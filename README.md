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


