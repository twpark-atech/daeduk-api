from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_URL = "triton:8000"

PATCH_SIZE = 256
MASK_THRESHOLD = 0.5
ORIGINAL_H, ORIGINAL_W = 11787, 10355

RPATH = "/app/data"
FEATURE_PATH = os.path.join(RPATH, "features_patched.h5")  # Docker에서 마운트해야 함
REF_PATH = os.path.join(RPATH, "impervious_5m_5186.tif")
LC_PATH = os.path.join(RPATH, "토지피복.shp")
RD_PATH = os.path.join(RPATH, "교량데이터.shp")

KMA_SFCTM3_URL = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php'
GETULTRASRTFCST_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'

KMA_SFCTM3_KEY = os.getenv("KMA_SFCTM3")
GETULTRASRTFCST_KEY = os.getenv("GETULTRASRTFCST")