#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define likely(_x)   __builtin_expect(!!(_x), 1)
#define unlikely(_x)  __builtin_expect(!!(_x), 0)

#define MAP_SIZE 65536

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

static u8 virgin_map[MAP_SIZE];
static u8 *trace_bits;

static void init_count_class16(void);
static inline void classify_counts(u64* mem);
static inline u32 hash32(const void* key, u32 len, u32 seed);

int init()
{
  int id, i;

  id = shmget(IPC_PRIVATE, MAP_SIZE, IPC_CREAT | IPC_EXCL | 0600);
  trace_bits = shmat(id, NULL, 0);

  memset(virgin_map, 0xff, MAP_SIZE);
  
  init_count_class16();
  
  return id;
}

int clear_trace() {
  memset(trace_bits, 0, MAP_SIZE);
  return 0;
}

static struct {
  u32 len;
  struct {
    u32 idx;
    u8 v;
  } values[MAP_SIZE];
} syndrome;

int has_new_bits() {
  u64* current = (u64*)trace_bits;
  u64* virgin  = (u64*)virgin_map;

  classify_counts(current);
  syndrome.len = 0;
  u32  i = (MAP_SIZE >> 3);
  int   ret = 0;
  u32 idx = 0;

  while (i--) {
    /* Optimize for (*current & *virgin) == 0 - i.e., no bits in current bitmap
       that have not been already cleared from the virgin map - since this will
       almost always be the case. */

    u32 j;
    u8* cur = (u8*)current;
    u8* vir = (u8*)virgin;
    for (j = 0; j < 8; j++) {
      u8 v = vir[j] & cur[j];
      if (v > 0) {
	syndrome.values[syndrome.len].idx = idx+j;
	syndrome.values[syndrome.len].v = v;
	syndrome.len++;
      }
    }
    
    if (unlikely(*current) && unlikely(*current & *virgin)) {

      if (likely(ret < 2)) {
        /* Looks like we have not found any new bytes yet; see if any non-zero
           bytes in current[] are pristine in virgin[]. */

        if ((cur[0] && vir[0] == 0xff) || (cur[1] && vir[1] == 0xff) ||
            (cur[2] && vir[2] == 0xff) || (cur[3] && vir[3] == 0xff) ||
            (cur[4] && vir[4] == 0xff) || (cur[5] && vir[5] == 0xff) ||
            (cur[6] && vir[6] == 0xff) || (cur[7] && vir[7] == 0xff)) ret = 2;
        else ret = 1;

      }

      *virgin &= ~*current;

    }

    idx += 8;
    current++;
    virgin++;

  }

  return ret;

}


int hash() {
  return hash32(trace_bits, MAP_SIZE, 0xa5b35705);
}

void *get_syndrome() {
  return &syndrome;
}

u8 *get_trace_bits() {
  return trace_bits;
}

static const u8 count_class_lookup8[256] = {

  [0]           = 0,
  [1]           = 1,
  [2]           = 2,
  [3]           = 4,
  [4 ... 7]     = 8,
  [8 ... 15]    = 16,
  [16 ... 31]   = 32,
  [32 ... 127]  = 64,
  [128 ... 255] = 128

};

static u16 count_class_lookup16[65536];

static void init_count_class16(void) {

  u32 b1, b2;

  for (b1 = 0; b1 < 256; b1++) 
    for (b2 = 0; b2 < 256; b2++)
      count_class_lookup16[(b1 << 8) + b2] = 
        (count_class_lookup8[b1] << 8) |
        count_class_lookup8[b2];

}

static inline void classify_counts(u64* mem) {

  u32 i = MAP_SIZE >> 3;

  while (i--) {

    /* Optimize for sparse bitmaps. */

    if (unlikely(*mem)) {

      u16* mem16 = (u16*)mem;

      mem16[0] = count_class_lookup16[mem16[0]];
      mem16[1] = count_class_lookup16[mem16[1]];
      mem16[2] = count_class_lookup16[mem16[2]];
      mem16[3] = count_class_lookup16[mem16[3]];

    }

    mem++;

  }

}


#define ROL64(_x, _r)  ((((u64)(_x)) << (_r)) | (((u64)(_x)) >> (64 - (_r))))

static inline u32 hash32(const void* key, u32 len, u32 seed) {

  const u64* data = (u64*)key;
  u64 h1 = seed ^ len;

  len >>= 3;

  while (len--) {

    u64 k1 = *data++;

    k1 *= 0x87c37b91114253d5ULL;
    k1  = ROL64(k1, 31);
    k1 *= 0x4cf5ad432745937fULL;

    h1 ^= k1;
    h1  = ROL64(h1, 27);
    h1  = h1 * 5 + 0x52dce729;

  }

  h1 ^= h1 >> 33;
  h1 *= 0xff51afd7ed558ccdULL;
  h1 ^= h1 >> 33;
  h1 *= 0xc4ceb9fe1a85ec53ULL;
  h1 ^= h1 >> 33;

  return h1;

}
