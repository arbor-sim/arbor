CC=clang++
FLAGS=-std=c++11 -g

test.exe : main.cpp *.hpp makefile
	$(CC) $(FLAGS) main.cpp -o test.exe

clean :
	rm -f test.exe
	rm -f a.out
