CC=clang++
FLAGS=-std=c++11 -g

test.exe : main.cpp *.hpp makefile gtest.o
	$(CC) $(FLAGS) main.cpp -o test.exe gtest.o -pthread

gtest.o :
	$(CC) $(FLAGS) gtest-all.cc -c -o gtest.o

clean :
	rm -f test.exe
	rm -f gtest.o
	rm -f a.out
