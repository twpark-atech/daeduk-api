from dotenv import load_dotenv
import os
from affine import Affine

load_dotenv()

CLIENT_URL = "triton:8000"
TRITON_SERVER_URL = "triton:8000"

PATCH_SIZE = 256
MASK_THRESHOLD = 0.6

COLOR = (0, 100, 255) # Blue
LINK_ID = "LINK_ID"
ALPHA = 255 # 투명도
BLOCK_SIZE = 4
USE_GAUSSIAN = True
SIGMA = 5

CRS = "EPSG:5186"

# ORIGINAL_H, ORIGINAL_W = 11787, 10355 # Busan
ORIGINAL_H, ORIGINAL_W = 11455, 13958 # Ulsan

# REF_HEIGHT, REF_WIDTH = 11787, 10355 # Busan
REF_HEIGHT, REF_WIDTH = 11520, 14080 # Ulsan
# TRANSFORM = Affine(5.00, 0.00, 359713.47,
#                    0.00, -5.00, 313851.57) # Busan
TRANSFORM = Affine(5.00, 0.00, 358208.3574,
                   0.00, -5.00, 353157.3257) # Ulsan

RPATH = "/app/data"
# MASK_PATH = os.path.join(RPATH, "union_array_5m.npy") # Busan
MASK_PATH = os.path.join(RPATH, "ulsan_union_mask_5m.npy") # Ulsan
# FEATURE_PATH = os.path.join(RPATH, "features_ulsan_patched.h5") # Busan
FEATURE_PATH = os.path.join(RPATH, "features_ulsan_patched.h5") # Ulsan
LINK_PATH = os.path.join(RPATH, "MOCT_LINK.shp")

KMA_SFCTM3_URL = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php'
GETULTRASRTFCST_URL = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'

KMA_SFCTM3_KEY = os.getenv("KMA_SFCTM3")
GETULTRASRTFCST_KEY = os.getenv("GETULTRASRTFCST")