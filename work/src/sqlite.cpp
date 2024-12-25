#include <sqlite_modern_cpp.h>

using namespace sqlite;
using namespace std;

// データベースを取得する
database get_db() {
  static const string db_path = "cache/data.db";
  static database db(db_path);
  return db;
}

// keyに対応するvalueを取得する
string get_cache(string cache_id, string key) {
  database db = get_db();
  string table_name = cache_id;

  try {
    string value;
    db << "SELECT value FROM " + table_name + " WHERE key = ?;" << key >> value;
    return value;
  } catch (sqlite_exception e) {
    return "";
  }
}

// keyに対応するvalueを保存する
bool save_cache(string cache_id, string key, string value) {
  database db = get_db();
  string table_name = cache_id;

  string query = "CREATE TABLE IF NOT EXISTS " + table_name +
                 " (key TEXT PRIMARY KEY, value TEXT);";

  try {
    db << "CREATE TABLE IF NOT EXISTS " + table_name +
              " (key TEXT PRIMARY KEY, value TEXT);";
    db << "INSERT OR REPLACE INTO " + table_name +
              " (key, value) VALUES (?, ?);"
       << key << value;
  } catch (sqlite_exception e) {
    cout << e.errstr() << endl;
    cout << e.what() << endl;
    exit(1);
  }
  return true;
}
