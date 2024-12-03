# Eigenインストール
[参考記事](https://qiita.com/nishiys/items/1585d26a824862eec36b)

## インストール
```
sudo apt install libeigen3-dev
```

## Include Pathについて
自分の環境では、
`/usr/include/`配下にインストールされていたため、
Makefileで`/usr/include/`をInclude Pathに指定することで解決した。


# OpenCVインストール
[参考記事](https://qiita.com/hsharaku/items/dde299aafafcfbfb8fca#opencv%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)
## インストール
```
sudo apt-get install libopencv-dev
```
## 確認
```
opencv_version
```

# 環境構築（vscode 拡張機能 clangd）
### compile_commands.jsonの生成
```
sudo apt-get update
sudo apt-get install bear
```
```
bear -- make
```
