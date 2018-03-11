#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

const int N = 100;

/*
字符串或者文本 编辑距离，对文本或者字符串进行三种操作：
　　　１．删除字符
　　　２．添加字符
　　　３．替换字符
假设我们有两个字符串：ａ和ｂ，显然可知，对字符串ａ插入和对字符串ｂ删除等价，反之也成立
                　　　　　　　　　那么我们可以不用考虑其中一种操作，删除或者添加，我们这里不考虑添加．
                　　　　　　　　　替换ａ的某一个字符或者替换ｂ的某一个字符是等价的

对于字符串ａ和ｂ，只考虑他们的最后一个字符；
　　　　　　　　　　　　　　　如果最后一个字符相同：edit_distance(a, b) = edit_distance(a - a_last_char, b - b_last_char)
               如果最后一个字符不同，可以有三种操作：
               　　１．替换ａ(或者ｂ)，此时edit_distance(a, b)　＝　edit_distance(a - a_last_char, b - b_last_char) + 1
               　　２．删除ａ最后一个字符，edit_distance(a, b)　＝　edit_distance(a - a_last_char, b) + 1
               　　３．删除ｂ最后一个字符，edit_distance(a, b)　＝　edit_distance(a, b - b_last_char) + 1
*/


int length(char **a){
    int len = 0;
    char *tem = a[0];
    while(tem){
        len++;
        tem = a[len];
    }
    return len;
}

int edit_distance_doc(char **a, char **b){
    int lena = length(a);
    int lenb = length(b);
    int distance[N][N] = {0};

    for(int i = 0; i <= lenb; ++i) distance[0][i] = i;
    for(int i = 0; i <= lena; ++i) distance[i][0] = i;

    for(int i = 1; i <= lena; ++i) {
        for(int j = 1; j <= lenb; ++j) {
            if(strcmp(a[i - 1], b[j - 1])){
                distance[i][j] = distance[i - 1][j - 1];
            } else {
                distance[i][j] = min(distance[i - 1][j], min(distance[i][j - 1], distance[i - 1][j - 1])) + 1;
            }
        }
    }
    return distance[lena][lenb];
}

int edit_distance_str(char* a, char* b) {
    int lena = strlen(a);
    int lenb = strlen(b);
    int distance[N][N] = {0};

    for(int i = 0; i < lenb; ++i) distance[0][i] = i;
    for(int i = 0; i < lena; ++i) distance[i][0] = i;

    for(int i = 1; i <= lena; ++i) {
        for(int j = 1; j <= lenb; ++j) {
            if(a[i - 1] == b[j - 1]){
                distance[i][j] = distance[i - 1][j - 1];
            } else {
                distance[i][j] = min(distance[i - 1][j], min(distance[i][j - 1], distance[i - 1][j - 1])) + 1;
            }
        }
    }
    return distance[lena][lenb];
}

extern "C" {
    int edit_distance_string(char *a, char *b) {
        return edit_distance_str(a, b);
    }

    int edit_distance_document(char **a, char **b) {
        return edit_distance_doc(a, b);
    }
}

// int main()
// {
//     char* a[10] = {"adsdfgsfdgfsdf", "dsfsdfsdfasdf", "dfasdfsdafasdf", "sdfasdfasddsafdsaf"};
//     char* b[10] = {"adsfsdfgsdfgsdfgsdfgdsdf", "dfasdfsdafasdf", "sdfasdfasddsafdsaf"};

//     printf("%d\n", edit_distance_document(a, b));
//     return 0;
// }