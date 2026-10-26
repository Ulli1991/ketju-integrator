// Ketju-integrator tests: GoogleTest stand-in.
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license.
// See LICENSE.md or <https://www.gnu.org/licenses/> for details.
//
// Minimal stand-in for the part of the GoogleTest API used by the KETJU unit tests (unit-tests/), so they compile
// unmodified without the googletest submodule: TEST, TEST_P + INSTANTIATE_TEST_SUITE_P with testing::Combine of two
// testing::Values lists, ASSERT_* / EXPECT_* (with << messages). The runner is main() in tests/gizmo_additions_kokkos.cpp.

#ifndef RUMI_GTEST_SHIM_H
#define RUMI_GTEST_SHIM_H

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace testing {

struct TestCase {
    std::string name;
    std::function<void()> run;
};
inline std::vector<TestCase> &registry() {
    static std::vector<TestCase> r;
    return r;
}
inline int &failures() {
    static int f = 0;
    return f;
}
struct Registrar {
    Registrar(const std::string &n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

struct AssertionAbort {};
inline void report(bool fatal, const char *file, int line, const std::string &msg);
struct Reporter {
    bool fatal;
    const char *file;
    int line;
    std::string msg;
    Reporter(bool f, const char *fl, int l, std::string m) : fatal(f), file(fl), line(l), msg(std::move(m)) {}
    template <typename M> void operator=(const M &m) {
        const std::string extra = m.str();
        report(fatal, file, line, extra.empty() ? msg : msg + "\n" + extra);
    }
};

template <typename P> class TestWithParam {
  public:
    using ParamType = P;
    virtual ~TestWithParam() = default;
    const P &GetParam() const { return *param_; }
    const P *param_ = nullptr;
    virtual void TestBody() = 0;
};

template <typename T> struct ValueList { std::vector<T> v; };
template <typename T, typename... Ts> ValueList<T> Values(T a, Ts... b) { return ValueList<T>{{a, T(b)...}}; }
template <typename A, typename B> std::vector<std::tuple<A, B>> Combine(ValueList<A> a, ValueList<B> b) {
    std::vector<std::tuple<A, B>> out;
    for (auto &x : a.v)
        for (auto &y : b.v) out.emplace_back(x, y);
    return out;
}
template <typename P> struct TestParamInfo {
    P param;
    size_t index;
};

// Parametrised tests: bodies registered per fixture, instantiated over a parameter list
template <typename Fixture> struct ParamBodies {
    static std::vector<std::pair<std::string, std::function<Fixture *()>>> &get() {
        static std::vector<std::pair<std::string, std::function<Fixture *()>>> b;
        return b;
    }
};
template <typename Fixture> struct ParamBodyRegistrar {
    ParamBodyRegistrar(const std::string &n, std::function<Fixture *()> make) {
        ParamBodies<Fixture>::get().push_back({n, std::move(make)});
    }
};
template <typename Fixture, typename List, typename Namer>
int instantiate(const std::string &prefix, const std::string &suite, List params, Namer namer) {
    using P = typename Fixture::ParamType;
    auto shared = std::make_shared<std::vector<P>>(params.begin(), params.end());
    for (auto &body : ParamBodies<Fixture>::get()) {
        for (size_t i = 0; i < shared->size(); ++i) {
            TestParamInfo<P> info{(*shared)[i], i};
            std::string name = prefix + "/" + suite + "." + body.first + "/" + namer(info);
            auto make = body.second;
            registry().push_back({name, [shared, i, make]() {
                                      Fixture *f = make();
                                      f->param_ = &(*shared)[i];
                                      try {
                                          f->TestBody();
                                      } catch (AssertionAbort &) {}
                                      delete f;
                                  }});
        }
    }
    return 0;
}

inline void InitGoogleTest(int *, char **) {}

// failure messages with << (gtest's AssertHelper = Message idiom)
class Message {
  public:
    template <typename T> Message &operator<<(const T &v) {
        s_ << v;
        return *this;
    }
    std::string str() const { return s_.str(); }

  private:
    std::ostringstream s_;
};
template <typename T> void put(std::ostringstream &s, const T &v) { s << v; }
inline void put(std::ostringstream &s, std::nullptr_t) { s << "nullptr"; }
template <typename A, typename B, typename F> std::string cmp_msg(const A &a, const B &b, F f, const char *what) {
    if (f(a, b)) return std::string();
    std::ostringstream s;
    s.precision(17);
    s << what;
    put(s, a);
    s << " vs ";
    put(s, b);
    return s.str();
}
inline std::string near_msg(double a, double b, double tol, const char *what) {
    if (std::fabs(a - b) <= tol) return std::string();
    std::ostringstream s;
    s.precision(17);
    s << what << std::fabs(a - b) << " > " << tol << " (" << a << " vs " << b << ")";
    return s.str();
}

inline void report(bool fatal, const char *file, int line, const std::string &msg) {
    std::fprintf(stderr, "%s:%d: Failure\n%s\n", file, line, msg.c_str());
    ++failures();
    if (fatal) throw AssertionAbort{};
}

} // namespace testing

#define RUMI_GT_CMP(fatal, a, b, op, opname)                                                                          \
    switch (0)                                                                                                        \
    case 0:                                                                                                           \
    default:                                                                                                          \
        if (const std::string rumi_m = ::testing::cmp_msg(                                                            \
                (a), (b), [](const auto &rumi_x, const auto &rumi_y) { return rumi_x op rumi_y; },                    \
                "Expected " #a " " opname " " #b ", got ");                                                           \
            rumi_m.empty())                                                                                           \
            ;                                                                                                         \
        else                                                                                                          \
            ::testing::Reporter(fatal, __FILE__, __LINE__, rumi_m) = ::testing::Message()
#define RUMI_GT_NEAR(fatal, a, b, tol)                                                                                \
    switch (0)                                                                                                        \
    case 0:                                                                                                           \
    default:                                                                                                          \
        if (const std::string rumi_m = ::testing::near_msg((a), (b), (tol), "|" #a " - " #b "| = ");                 \
            rumi_m.empty())                                                                                           \
            ;                                                                                                         \
        else                                                                                                          \
            ::testing::Reporter(fatal, __FILE__, __LINE__, rumi_m) = ::testing::Message()

#define ASSERT_EQ(a, b) RUMI_GT_CMP(true, a, b, ==, "==")
#define ASSERT_NE(a, b) RUMI_GT_CMP(true, a, b, !=, "!=")
#define ASSERT_LT(a, b) RUMI_GT_CMP(true, a, b, <, "<")
#define ASSERT_LE(a, b) RUMI_GT_CMP(true, a, b, <=, "<=")
#define ASSERT_GT(a, b) RUMI_GT_CMP(true, a, b, >, ">")
#define ASSERT_GE(a, b) RUMI_GT_CMP(true, a, b, >=, ">=")
#define ASSERT_TRUE(a) RUMI_GT_CMP(true, (bool)(a), true, ==, "==")
#define ASSERT_FALSE(a) RUMI_GT_CMP(true, (bool)(a), false, ==, "==")
#define EXPECT_EQ(a, b) RUMI_GT_CMP(false, a, b, ==, "==")
#define EXPECT_NE(a, b) RUMI_GT_CMP(false, a, b, !=, "!=")
#define EXPECT_LT(a, b) RUMI_GT_CMP(false, a, b, <, "<")
#define EXPECT_LE(a, b) RUMI_GT_CMP(false, a, b, <=, "<=")
#define EXPECT_GT(a, b) RUMI_GT_CMP(false, a, b, >, ">")
#define EXPECT_GE(a, b) RUMI_GT_CMP(false, a, b, >=, ">=")
#define EXPECT_TRUE(a) RUMI_GT_CMP(false, (bool)(a), true, ==, "==")
#define EXPECT_FALSE(a) RUMI_GT_CMP(false, (bool)(a), false, ==, "==")
#define ASSERT_NEAR(a, b, t) RUMI_GT_NEAR(true, a, b, t)
#define EXPECT_NEAR(a, b, t) RUMI_GT_NEAR(false, a, b, t)

#define TEST(suite, name)                                                                                             \
    static void rumi_test_##suite##_##name();                                                                         \
    static ::testing::Registrar rumi_reg_##suite##_##name(#suite "." #name, [] {                                      \
        try {                                                                                                         \
            rumi_test_##suite##_##name();                                                                             \
        } catch (::testing::AssertionAbort &) {}                                                                      \
    });                                                                                                               \
    static void rumi_test_##suite##_##name()

#define TEST_P(fixture, name)                                                                                         \
    struct rumi_tp_##fixture##_##name : public fixture {                                                              \
        void TestBody() override;                                                                                     \
    };                                                                                                                \
    static ::testing::ParamBodyRegistrar<fixture> rumi_tpreg_##fixture##_##name(                                      \
        #name, []() -> fixture * { return new rumi_tp_##fixture##_##name; });                                         \
    void rumi_tp_##fixture##_##name::TestBody()

#define INSTANTIATE_TEST_SUITE_P(prefix, fixture, params, ...)                                                        \
    static int rumi_inst_##prefix##_##fixture =                                                                       \
        ::testing::instantiate<fixture>(#prefix, #fixture, params, __VA_ARGS__)

#endif // RUMI_GTEST_SHIM_H
