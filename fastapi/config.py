from dotenv import load_dotenv
import os
from affine import Affine

load_dotenv()

CLIENT_URL = "triton:8000"

PATCH_SIZE = 256
MASK_THRESHOLD = 0.5
ORIGINAL_H, ORIGINAL_W = 11787, 10355

REF_HEIGHT = 11787
REF_WIDTH = 10355
TRANSFORM = Affine(5.00, 0.00, 359713.47,
                   0.00, -5.00, 313851.57)
CRS = "EPSG:5186"

RPATH = "/app/data"
MASK_PATH = os.path.join(RPATH, "union_array_5m.npy")
FEATURE_PATH = os.path.join(RPATH, "features_patched.h5")

KMA_SFCTM3_URL = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php'
GETULTRASRTFCST_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'

KMA_SFCTM3_KEY = os.getenv("KMA_SFCTM3")
GETULTRASRTFCST_KEY = os.getenv("GETULTRASRTFCST")