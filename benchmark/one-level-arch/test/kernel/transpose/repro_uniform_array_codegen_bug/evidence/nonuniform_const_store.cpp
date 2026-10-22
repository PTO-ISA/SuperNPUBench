// Control: non-uniform constants produce the correct codegen.
extern "C" volatile unsigned sink[4];
int main() {
    constexpr unsigned s[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) {
        sink[i] = s[i];
    }
    return 0;
}
