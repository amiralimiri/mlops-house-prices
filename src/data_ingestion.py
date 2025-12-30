"""
This module handles downloading CSV datasets (train & test),
splitting train into train/validation sets, and saving them as CSV files.
"""

from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.data_ingestion_config = config["data_ingestion"]

        self.bucket_name = self.data_ingestion_config["bucket_name"]
        self.storage_path = self.data_ingestion_config["storage_path"]

        self.object_name_train = self.data_ingestion_config[
            "object_name_train"
        ]  # train.csv
        self.object_name_test = self.data_ingestion_config[
            "object_name_test"
        ]  # test.csv

        # split configs
        self.val_ratio = self.data_ingestion_config.get("val_ratio", 0.2)
        self.random_state = self.data_ingestion_config.get("random_state", 42)
        self.shuffle = self.data_ingestion_config.get("shuffle", True)

        # target handling
        self.target_column = self.data_ingestion_config.get(
            "target_column", "SalePrice"
        )
        self.dropna_target = self.data_ingestion_config.get("dropna_target", True)

        self.train_url = (
            f"https://{self.bucket_name}.{self.storage_path}/{self.object_name_train}"
        )
        self.test_url = (
            f"https://{self.bucket_name}.{self.storage_path}/{self.object_name_test}"
        )

        artifact_dir = Path(self.data_ingestion_config["artifact_dir"])
        self.raw_dir = artifact_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def download_csv(self, url: str) -> pd.DataFrame:
        """Download CSV from a URL and return it as a DataFrame."""
        try:
            with urlopen(url) as response:
                csv_text = response.read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_text))
            logger.info(f"Downloaded {url} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to download or parse CSV from {url}: {str(e)}")
            raise

    def split_train_data(self, df: pd.DataFrame):
        """Split train dataframe into train and validation sets."""
        if self.dropna_target and self.target_column in df.columns:
            before = len(df)
            df = df.dropna(subset=[self.target_column])
            after = len(df)
            if before != after:
                logger.info(
                    f"Dropped {before - after} rows with missing target '{self.target_column}'"
                )

        train_df, val_df = train_test_split(
            df,
            test_size=self.val_ratio,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        logger.info(f"Train split: {train_df.shape}")
        logger.info(f"Validation split: {val_df.shape}")

        return train_df, val_df

    def save_to_csv_files(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """Save train, validation, and test dataframes to CSV files."""
        paths = {
            "train": self.raw_dir / "train.csv",
            "validation": self.raw_dir / "validation.csv",
            "test": self.raw_dir / "test.csv",
        }

        train_df.to_csv(paths["train"], index=False)
        val_df.to_csv(paths["validation"], index=False)
        test_df.to_csv(paths["test"], index=False)

        logger.info(f"Saved train set: {paths['train']} ({len(train_df)} rows)")
        logger.info(f"Saved validation set: {paths['validation']} ({len(val_df)} rows)")
        logger.info(f"Saved test set: {paths['test']} ({len(test_df)} rows)")

    def run(self):
        """
        Execute the data ingestion pipeline:
        1. Download train.csv
        2. Split into train / validation
        3. Download test.csv
        4. Save all datasets
        """
        logger.info("Data Ingestion started")
        logger.info(f"Train source: {self.train_url}")
        logger.info(f"Test source: {self.test_url}")

        # Load data
        train_df_full = self.download_csv(self.train_url)
        test_df = self.download_csv(self.test_url)

        # Split train
        train_df, val_df = self.split_train_data(train_df_full)

        # Save outputs
        self.save_to_csv_files(train_df, val_df, test_df)

        logger.info("Data Ingestion completed successfully")
