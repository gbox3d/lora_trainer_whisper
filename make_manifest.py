import json
import re
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="데이터셋 매니페스트 생성 스크립트")
    parser.add_argument("--root", type=str, default="datasets/Sample", help="데이터셋 루트 경로")
    parser.add_argument("--wav_dir", type=str, default="wav", help="오디오 폴더명")
    parser.add_argument("--label_dir", type=str, default="lb", help="라벨 폴더명")
    parser.add_argument("--output", type=str, default="manifest.jsonl", help="출력 파일명")
    return parser.parse_args()

def normalize_text(t: str) -> str:
    # (A)/(B) -> B 패턴 처리
    t = re.sub(r"\(([^()]+?)\)/\(([^()]+?)\)", r"\2", t)
    # 남은 괄호 제거
    t = t.replace("(", "").replace(")", "")
    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    args = parse_args()
    
    root = Path(args.root)
    wav_root = root / args.wav_dir
    lab_root = root / args.label_dir
    out_path = root / args.output

    if not lab_root.exists():
        print(f"❌ Error: 라벨 폴더를 찾을 수 없습니다: {lab_root}")
        return

    print(f"📂 Root: {root}")
    print(f"   Searching labels in: {lab_root}")
    print(f"   Matching wavs in:    {wav_root}")

    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        # rglob을 사용하여 하위 폴더까지 탐색
        for jp in lab_root.rglob("*.json"):
            try:
                obj = json.loads(jp.read_text(encoding="utf-8"))
                
                # JSON 구조에 따라 script 키가 없을 수도 있으므로 예외처리
                if "script" not in obj or "text" not in obj["script"]:
                    continue
                    
                text = normalize_text(obj["script"]["text"])

                # 라벨 파일의 상대 경로를 이용해 wav 경로 추론
                rel = jp.relative_to(lab_root).with_suffix(".wav")
                wav_path = wav_root / rel
                
                if not wav_path.exists():
                    # print(f"⚠️ Missing audio: {wav_path}") # 디버깅 시 주석 해제
                    continue

                out.write(json.dumps({"audio": str(wav_path), "text": text}, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"Error processing {jp}: {e}")
                continue

    print("-" * 30)
    print(f"✅ Created: {out_path}")
    print(f"📊 Total Rows: {count}")

if __name__ == "__main__":
    main()