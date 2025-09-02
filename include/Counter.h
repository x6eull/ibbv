#include <iostream>
#include <map>
#include <ostream>
#include <tuple>
#include <utility>

template <typename... T>
std::ostream &operator<<(std::ostream &stream, const std::tuple<T...> &t) {
  stream << "(";
  std::apply(
      [&](const auto &...args) {
        size_t i = 0;
        ((stream << args << (++i < sizeof...(T) ? ", " : "")), ...);
      },
      t);
  stream << ")";
  return stream;
}

namespace ibbv::utils {
template <typename... T> class Counter {
public:
  using key_t = std::tuple<T...>;
  void inc(const key_t &key) {
    const auto it = recMap.find(key);
    if (it == recMap.end())
      recMap.emplace(std::make_pair(key, 1));
    else
      it->second++;
  }
  template <typename... Args> void inc(const Args &...args) {
    inc(std::make_tuple(std::forward<Args>(args)...));
  }
  Counter(const std::string &title) : title(title) {}
  ~Counter() {
    for (const auto [k, c] : recMap)
      std::cout << "[" << title << "] " << k << ": " << c << std::endl;
  }

protected:
  std::string title;
  std::map<key_t, int> recMap;
};
} // namespace ibbv::utils