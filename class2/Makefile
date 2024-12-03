# コンパイラ
CXX = g++

# コンパイルフラグ
CXXFLAGS = -std=c++17 -O2 -I /usr/include -I /usr/include/opencv4 
LIBS = `pkg-config --cflags --libs opencv4`

# ターゲット実行ファイル名と出力ディレクトリ
OUT_DIR = out
TARGET = $(OUT_DIR)/app

# ソースファイルとオブジェクトファイル
SRCS = $(wildcard src/*.cpp)  # srcディレクトリからソースファイルを取得
OBJS = $(SRCS:src/%.cpp=$(OUT_DIR)/%.o)  # 出力先もsrcディレクトリを考慮して修正

# デフォルトターゲット
all: init run

# 実行ファイルを生成
$(TARGET): $(OBJS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# .cppから.oを生成
$(OUT_DIR)/%.o: src/%.cpp  # ソースファイルのパスをsrcに変更
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build: $(TARGET)

# 実行ターゲット
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)
	
# outputディレクトリがなければ作成
init:
	mkdir -p output

# クリーンアップ
clean:
	rm -rf $(OUT_DIR)
	rm -rf output

# ファイル一覧を表示するデバッグターゲット
debug:
	@echo "SRCS: $(SRCS)"
	@echo "OBJS: $(OBJS)"
	@echo "HEADERS: $(wildcard src/*.h)"  # src内のヘッダファイルを表示
