# コンパイラ
CXX = g++

# コンパイルフラグ
CXXFLAGS = -std=c++17 -O2 -I /usr/include
CXXFLAGS += -I /usr/include/opencv4 -I /usr/include/eigen3 
CXXFLAGS += -I /workspaces/libs/sqlite_modern_cpp/hdr

LIBS = `pkg-config --cflags --libs opencv4`
LIBS += -lsqlite3
GOLD = -fuse-ld=gold

# ターゲット実行ファイル名と出力ディレクトリ
OUT_DIR = out
TARGET = $(OUT_DIR)/app

# ソースファイルとオブジェクトファイル
SRCS = $(wildcard src/*.cpp)  # srcディレクトリからソースファイルを取得
OBJS = $(SRCS:src/%.cpp=$(OUT_DIR)/%.o)  # 出力先もsrcディレクトリを考慮して修正

# デフォルトターゲット
all: 
	make start -j6
	
start: init run
	
# 実行ファイルを生成
$(TARGET): $(OBJS)
	@mkdir -p $(OUT_DIR)
	@echo "Linking $@..."
	@$(CXX) $(CXXFLAGS) $(GOLD) -o $@ $^ $(LIBS) 

# .cppから.oを生成
$(OUT_DIR)/%.o: src/%.cpp  # ソースファイルのパスをsrcに変更
	@mkdir -p $(OUT_DIR)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

build: $(TARGET)

# 実行ターゲット
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)
	
# outputディレクトリがなければ作成
init:
	mkdir -p output
	mkdir -p cache
	mkdir -p pydata

# クリーンアップ
clean:
	rm -rf $(OUT_DIR)
	rm -rf output
	rm -rf pydata
	
# キャッシュのクリーンアップ
cache_clean:
	rm -rf cache

# ファイル一覧を表示するデバッグターゲット
debug:
	@echo "SRCS: $(SRCS)"
	@echo "OBJS: $(OBJS)"
	@echo "HEADERS: $(wildcard src/*.h)"  # src内のヘッダファイルを表示

# bear でコンパイルコマンドを生成
# VSCodeの静的解析のために必要
compile_commands: 
	make clean
	@echo ""
	bear -- make

plot:
	python3 python/main.py
