import os
import sqlite3
import hashlib
import pickle

DB_DIR = "cache/"


def sqlite_cache(db_path="cache.db"):
    # もしDB_PATHのディレクトリが存在しない場合は作成
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)

    def decorator(func):
        # データベース初期化
        conn = sqlite3.connect(DB_DIR + db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        """)
        conn.commit()

        def wrapper(*args, **kwargs):
            # キャッシュキーを生成
            key = hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()

            # キャッシュをチェック
            cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                print("Cache hit!")
                return pickle.loads(row[0])

            # 関数を実行して結果を保存
            print("Cache miss!")
            result = func(*args, **kwargs)
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, pickle.dumps(result))
            )
            conn.commit()
            return result

        return wrapper

    return decorator
