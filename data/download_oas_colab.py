"""download_oas_colab.py — Google Colab 에서 OAS **paired**(heavy+light 쌍) 항체 서열을
다운로드 → VH/VL V-도메인 AA 추출 → Google Drive 저장.

compute 노드는 firewall(외부망 차단)이라 OAS 직접 다운로드 불가 → Colab 에서 받아 Drive 로,
이후 Drive→compute 노드로 복사. 저장된 compact CSV 를 build_pssm.py(--refs) 가 그대로 먹는다.
(paired 라도 PSSM 은 VH/VL 각각의 위치별 분포만 쓰므로, 한 쌍 → H행 + L행 2개로 풀어서 저장.)

────────────────────────────────────────────────────────────────────────────
사용법 (Colab 셀 단위):
  1. OAS **paired** 검색: https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/
     필터(Species=human 등) → Search → "Download" → bulk_download.sh 받기.
  2. bulk_download.sh 를 Colab 에 업로드(좌측 파일창) → CELL 2 의 SH_PATH 지정.
     (또는 그 안 .csv.gz URL 들을 OAS_URLS 에 직접 붙여넣기.)
  3. 셀 순서대로 실행 → Drive 의 OUT_CSV 생성.
  4. compute 노드: cp "Drive/.../oas_paired_vdomains.csv" /home/aidx/DB/AGAB_MSADB/
  5. build_pssm.py --refs .../oas_paired_vdomains.csv ... 로 PSSM 재구축.
────────────────────────────────────────────────────────────────────────────
"""

# ===== CELL 1: Drive 마운트 =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 2: 설정 =====
import os, gzip, io, csv, json, random, re

OUT_DIR = '/content/drive/MyDrive/oas'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, 'oas_paired_vdomains.csv')
MAX_PER_UNIT = 50000      # unit 당 최대 *쌍* (paired 는 unit 이 큼 → subsample)
TOTAL_CAP    = 300000     # 전체 *쌍* 상한 (→ 서열은 최대 2배). PSSM 엔 충분
random.seed(0)

# (A) bulk_download.sh 업로드 후 경로 지정 (권장)
SH_PATH = '/content/bulk_download.sh'   # 없으면 '' 로
# (B) 또는 URL 직접 붙여넣기
OAS_URLS = [
    # "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/.../xxxx.csv.gz",
]
if SH_PATH and os.path.exists(SH_PATH):
    OAS_URLS = re.findall(r'(https?://\S+\.csv\.gz)', open(SH_PATH).read())

assert OAS_URLS, ("OAS_URLS 가 비었습니다! OAS paired 웹에서 bulk_download.sh 를 받아 "
                  "SH_PATH 로 지정하거나 URL 을 OAS_URLS 에 붙여넣으세요.")
print(f"다운로드 대상 paired unit: {len(OAS_URLS)}")

# ===== CELL 3: 다운로드 + VH/VL V-도메인 AA 추출 (paired 포맷) =====
# paired OAS unit(csv.gz): 1행=메타데이터(JSON), 2행=헤더, 이후 데이터.
# 컬럼은 _heavy / _light 접미사. V-도메인 = fwr1..fwr4 + cdr1..3 (gap 없음).
import urllib.request

AA = set('ARNDCQEGHILKMFPSTWYV')
REGION = ['fwr1_aa', 'cdr1_aa', 'fwr2_aa', 'cdr2_aa', 'fwr3_aa', 'cdr3_aa', 'fwr4_aa']


def extract_vdomain(row, suffix):
    """row 에서 한 chain(_heavy/_light) V-도메인 AA 재구성. region 없으면 alignment 폴백."""
    parts = [(row.get(c + suffix, '') or '') for c in REGION]
    seq = ''.join(parts).replace('-', '').replace('.', '').upper().strip()
    if len(seq) < 70:   # region 컬럼 없거나 빈 경우 → alignment 폴백
        aln = (row.get('sequence_alignment_aa' + suffix, '') or '')
        seq = aln.replace('-', '').replace('.', '').upper().strip()
    return seq if (70 <= len(seq) <= 200 and set(seq) <= AA) else None


written_pairs = 0
col_checked = False
with open(OUT_CSV, 'w', newline='') as fout:
    w = csv.writer(fout)
    w.writerow(['vdomain_aa', 'locus', 'v_call', 'pair_id'])
    for ui, url in enumerate(OAS_URLS):
        if written_pairs >= TOTAL_CAP:
            break
        try:
            raw = urllib.request.urlopen(url, timeout=180).read()
            with gzip.open(io.BytesIO(raw), 'rt') as gz:
                gz.readline()                          # 메타데이터 JSON 스킵
                reader = csv.DictReader(gz)
                if not col_checked:                    # 첫 unit 컬럼 점검(디버그)
                    has_h = any(c.endswith('_heavy') for c in (reader.fieldnames or []))
                    has_l = any(c.endswith('_light') for c in (reader.fieldnames or []))
                    print(f"  컬럼 점검: _heavy={has_h} _light={has_l} "
                          f"(예: {[c for c in (reader.fieldnames or []) if 'cdr3' in c][:2]})")
                    col_checked = True
                rows = list(reader)
            if len(rows) > MAX_PER_UNIT:
                rows = random.sample(rows, MAX_PER_UNIT)
            for r in rows:
                if written_pairs >= TOTAL_CAP:
                    break
                vh = extract_vdomain(r, '_heavy')
                vl = extract_vdomain(r, '_light')
                if vh and vl:                          # 쌍 둘 다 유효할 때만
                    pid = f"u{ui}_{written_pairs}"
                    w.writerow([vh, r.get('locus_heavy', 'IGH'), r.get('v_call_heavy', ''), pid])
                    w.writerow([vl, r.get('locus_light', ''), r.get('v_call_light', ''), pid])
                    written_pairs += 1
            print(f"[{ui+1}/{len(OAS_URLS)}] {url.split('/')[-1]}: 누적 쌍 {written_pairs}")
        except Exception as e:
            print(f"[{ui+1}] 실패 {url}: {e}")

print(f"\n완료: {written_pairs} 쌍 (= {written_pairs*2} 서열) → {OUT_CSV}")

# ===== CELL 4: 확인 =====
import pandas as pd
df = pd.read_csv(OUT_CSV)
print(df.shape)
print(df['locus'].value_counts())
print(df.head())
# → 이 CSV 를 compute 노드로 복사:  cp .../oas_paired_vdomains.csv /home/aidx/DB/AGAB_MSADB/
#   build_pssm.py 가 'vdomain_aa' 컬럼을 자동 인식해 처리 (리더는 노드에서 연결).
