from typing import *
import os
import re
import pickle

import tqdm
import hydra
import torch
from omegaconf import DictConfig
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split


import logging

logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )



AVAILABLE_MODELS = [
    "allegro/herbert-base-cased"
]



class TextDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.data.iloc[index]
        y = sample.y
        with open(sample.path, 'rb') as f:
            x = torch.from_numpy(np.load(f))

        return {
            'x': x.view(-1),
            'y': y,
            'text': sample.text
        }


class Datamodule(pl.LightningDataModule):

    def __init__(
        self, 
        data_path: str,
        target_path: str,
        embeddings_path: str,
        force_reprocess: bool = False,
        language_model: str = 'labse',
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        test_size: float = 0.2,
        train_val_ratio: float = 0.8,
        batch_size: int = 16,
        *args, 
        **kwargs
    ) -> None:
        super(Datamodule, self).__init__()
        
        self.data_path = data_path
        self.target_path = target_path
        self.embeddings_path = embeddings_path
        self.force_reproces = force_reprocess
        
        # language model 
        self.language_model = language_model
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # splits
        self.test_size = test_size
        self.train_val_ratio = train_val_ratio

        # training
        self.batch_size = batch_size


    def _get_decade(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[~df['date'].isna()].reset_index(drop=True)
        df['decade'] = df['date'].apply(lambda x: x.year // 10 * 10)
        df = df.reset_index(drop=True)
        return df
    
    def _map_decade(self, df: pd.DataFrame):
        decade_map = {d: i for i, d in enumerate(sorted(pd.unique(df.decade), reverse=True))}
        df['y'] = df.decade.apply(lambda x: decade_map[x])
        return df
    
    # EMBEDDINGS RELATED

    def _average_pool(
        self, 
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embeddings_transformer(
            self,
            df: pd.DataFrame
        ) -> None:
        pass
    
    def _get_embeddings(
            self, 
            df: pd.DataFrame
        ) -> pd.DataFrame:

        # instantiate tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        model = AutoModel.from_pretrained(self.language_model)

        paths = []
        i = 0
        for text in tqdm.tqdm(df.texts):
            output_path = os.path.join(self.embeddings_path, f'{i}.npy')
            if not os.path.exists(output_path):
                batch_dict = tokenizer(text, max_length=self.max_length, padding=self.padding, truncation=self.truncation, return_tensors='pt')
                outputs = model(**batch_dict)
                embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
                with open(output_path, 'wb') as f:
                    np.save(f, F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy())
            paths.append(output_path)

            i += 1
        
        df = pd.DataFrame({
            'text': df.texts,
            'path': paths,
            'decade': df.decade
        })

        return df

    def _get_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        train, test = train_test_split(df, test_size=self.test_size, stratify=df.decade)
        train, val = train_test_split(train, test_size=(1 - self.train_val_ratio), stratify=train.decade)


        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

        df = pd.concat([train, val, test])
        return df


    def _create_data(self) -> None:
        with open(self.data_path, 'rb') as f:
            df = pickle.load(f)

        logging.info(f'Getting the decade...')
        df = self._get_decade(df)
        logging.info(f'Done!')
        logging.info(f'Getting the embeddings. This may take a while...')
        df = self._get_embeddings(df)
        logging.info(f'Done!')
        logging.info(f'Creating splits...')
        df = self._get_splits(df)
        logging.info(f'Done!')
        logging.info(f'Mapping the decade...')
        df = self._map_decade(df)
        logging.info(f'Done!')

        df.to_csv(self.target_path, index=False)
        logging.info(f'Saved model to {self.target_path}')

    def prepare_data(self) -> None:
        if not os.path.exists(self.target_path) or self.force_reproces:
            logging.info(f'Dataset does not exist. Creating one...')
            self._create_data()
            
        self.data = pd.read_csv(self.target_path)
        logging.info(f'Loaded the data from {self.target_path}')
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TextDataset(data=self.data[self.data.split == 'train']),
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TextDataset(data=self.data[self.data.split == 'val']),
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            TextDataset(data=self.data[self.data.split == 'test']),
            batch_size=self.batch_size,
            shuffle=False
        )


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def _test(config: DictConfig) -> None:
    """
    Quick check, whether the model works
    """
    d = hydra.utils.instantiate(config.datamodule, _recursive_=False)
    d.prepare_data()
    print(d.train_dataloader())
    print(d.val_dataloader())
    print(d.test_dataloader())
    print(d.data.decade.nunique())
    print(d.train_dataloader().dataset[0]['x'].shape)


if __name__ == "__main__":
    _test()
