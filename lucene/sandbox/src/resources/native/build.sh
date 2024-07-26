g++ -g -x c -fPIC -c -I . -march=armv8-a ipbytebin.cpp -o ipbytebin.o 
g++ -shared -Wl,-install_name,libipbytebin.dylib -fvisibility=hidden -o libipbytebin.dylib ipbytebin.o
