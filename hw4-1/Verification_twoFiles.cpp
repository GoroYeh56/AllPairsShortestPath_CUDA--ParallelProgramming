// g++ -Wall -Wextra equifile.cpp -o equifile.exe

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <fstream>
using std::ios;
using std::ifstream;

#include <exception>
using std::exception;

#include <cstring>
#include <cstdlib>
using std::exit;
using std::memcmp;

bool equalFiles(ifstream& in1, ifstream& in2);

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        cerr << "Usage: equifile.exe <file1> <file2>" << endl;
        exit(-1);
    }

    try {
        ifstream in1(argv[1], ios::binary);
        ifstream in2(argv[2], ios::binary);

        if(equalFiles(in1, in2)) {
            cout << "Files are equal" << endl;
            exit(0);
        }
        else
        {
            cout << "Files are not equal" << endl;
            exit(1);
        }

    } catch (const exception& ex) {
        cerr << ex.what() << endl;
        exit(-2);
    }

    return -3;
}

bool equalFiles(ifstream& in1, ifstream& in2)
{
    ifstream::pos_type size1, size2;

    size1 = in1.seekg(0, ifstream::end).tellg();
    in1.seekg(0, ifstream::beg);

    size2 = in2.seekg(0, ifstream::end).tellg();
    in2.seekg(0, ifstream::beg);

    if(size1 != size2)
        return false;

    static const size_t BLOCKSIZE = 4096;
    size_t remaining = size1;

    while(remaining)
    {
        char buffer1[BLOCKSIZE], buffer2[BLOCKSIZE];
        size_t size = std::min(BLOCKSIZE, remaining);

        in1.read(buffer1, size);
        in2.read(buffer2, size);

        if(0 != memcmp(buffer1, buffer2, size)){
            std::cout<<"remain: "<<remaining<<std::endl;
            std::cout<<"My answer: "<<buffer1[size-1]<<std::endl;
            std::cout<<"Correct answer should be: "<<buffer2[size-1]<<std::endl;
            return false;
        }
        remaining -= size;
    }

    return true;
}