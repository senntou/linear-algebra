# コンパイラ
CXX = g++

# コンパイルフラグ
# CXXFLAGS = -Wall -Wextra -std=c++17 -O2
CXXFLAGS = -std=c++17 -O2 -I /usr/include

# ターゲット実行ファイル名と出力ディレクトリ
OUT_DIR = out
TARGET = $(OUT_DIR)/app

# ソースファイルとオブジェクトファイル
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
OBJS := $(patsubst %.o,$(OUT_DIR)/%.o,$(OBJS))

# デフォルトターゲット
all: run

# 実行ファイルを生成
$(TARGET): $(OBJS)
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# .cppから.oを生成
$(OUT_DIR)/%.o: %.cpp
	@mkdir -p $(OUT_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
build: $(TARGET)

# 実行ターゲット
run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

# クリーンアップ
clean:
	rm -rf $(OUT_DIR)

# ファイル一覧を表示するデバッグターゲット
debug:
	@echo "SRCS: $(SRCS)"
	@echo "OBJS: $(OBJS)"
