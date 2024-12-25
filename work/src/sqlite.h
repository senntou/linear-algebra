#include <sqlite_modern_cpp.h>

using namespace sqlite;
using namespace std;

// データベースを取得する
database get_db();

// keyに対応するvalueを取得する
string get_cache(string cache_id, string key);

// keyに対応するvalueを保存する
bool save_cache(string cache_id, string key, string value);
