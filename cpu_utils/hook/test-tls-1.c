 __thread int a[1028];
static  __thread int index = 0;

int fun1() {
  return a[index];
}
