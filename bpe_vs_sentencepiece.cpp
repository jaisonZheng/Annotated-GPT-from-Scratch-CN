#include <iostream>
#include <string>

int main() {
    // UTF-8 字符串
    std::string s = "你"; 
    
    // 输出大小：通常是 3 (bytes)
    std::cout << "Bytes size: " << s.size() << std::endl; 
    
    // 如果你要处理 Code Point，在现代 C++ 中通常用 char32_t
    std::u32string s32 = U"你";
    
    // 输出大小：1 (个 Code Point)
    std::cout << "Code Point count: " << s32.size() << std::endl; 
    
    return 0;
}