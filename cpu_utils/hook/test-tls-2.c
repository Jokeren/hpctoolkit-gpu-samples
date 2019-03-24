 __thread int b[1028];
static __thread int index = 0;

int fun2() {
  return b[index];
}
