from jinja_env import env2


class ModelBase:
    def generate_test_case(self, test_case_name):
        raise NotImplementedError("not implemented")

    def generate_test_cases(self, test_name, num_tests):
        for i in range(num_tests):
            yield self.generate_test_case(f"{test_name}_{i}")

    def render_files(self, test_name, num_tests=5, const_fname=None, src_fname=None):
        if const_fname is None:
            const_fname = f"constants_{test_name}.hpp"
        if src_fname is None:
            src_fname = f"test_{test_name}.cpp"
        cases = self.generate_test_cases(test_name, num_tests)
        const_snippets = []
        test_snippets = []
        for ts, cs in cases:
            const_snippets.extend(cs)
            test_snippets.append(ts)
        with open(const_fname, "w") as fid:
            print(f"generating {const_fname}")
            fid.write(
                env2.get_template("const_container.hpp").render(
                    constants=const_snippets, constants_header=const_fname
                )
            )
        with open(src_fname, "w") as fid:
            print(f"generating {src_fname}")
            fid.write(
                env2.get_template("gtest_container.cpp").render(
                    constants_header=const_fname,
                    using_directives=[],
                    tests=test_snippets,
                )
            )
