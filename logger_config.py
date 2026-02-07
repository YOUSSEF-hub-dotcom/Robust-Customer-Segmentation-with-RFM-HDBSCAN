import logging

def setup_logging():
    # منع التكرار لو الفانكشن اتندت أكتر من مرة
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("segmentation_pipeline.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )