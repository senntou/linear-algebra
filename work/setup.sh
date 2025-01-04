#!bin/bash

sudo apt-get update
sudo apt-get install -y sqlite3 libsqlite3-dev
sudo git clone https://github.com/SqliteModernCpp/sqlite_modern_cpp ./libs/sqlite_modern_cpp
