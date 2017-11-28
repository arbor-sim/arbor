#include "test.hpp"
#include "module.hpp"
#include "io/bulkio.hpp"

TEST(Module, open) {
    Module m(io::read_all(DATADIR "/test.mod"), "test.mod");
    if(!m.buffer().size()) {
        std::cout << "skipping Module.open test because unable to open input file" << std::endl;
        return;
    }
    Lexer lexer(m.buffer());
    auto t = lexer.parse();
    while(t.type != tok::eof) {
        t = lexer.parse();
        EXPECT_NE(t.type, tok::reserved);
    }
}
