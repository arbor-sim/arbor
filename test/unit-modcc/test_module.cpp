#include "common.hpp"
#include "io/bulkio.hpp"
#include "module.hpp"

TEST(Module, open) {
    Module m(io::read_all(DATADIR "/mod_files/test0.mod"), "test0.mod");
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

TEST(Module, linear_mechanisms) {
    for(int i = 1; i < 6; i++)
    {
        auto file_name = "test"+std::to_string(i)+".mod";

        Module m(io::read_all(DATADIR "/mod_files/" + file_name), file_name);
        if (!m.buffer().size()) {
            std::cout << "skipping Module.open test because unable to open input file" << std::endl;
            return;
        }

        Parser p(m, false);
        if (!p.parse()) {
            std::cout << "problem with parsing input file" << std::endl;
            return;
        }

        m.semantic();

        if(i < 3) {
            EXPECT_TRUE(m.is_linear());
        }
        else {
            EXPECT_FALSE(m.is_linear());
        }
    }
}
