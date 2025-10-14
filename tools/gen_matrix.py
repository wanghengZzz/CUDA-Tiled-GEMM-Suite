import random
import argparse


def gen_matrix(w, h, o, o_k, t_w, t_h, max_val, min_val, out_pth):

    
    matrix_A = [[random.randint(min_val, max_val) for _ in range(o) ] for _ in range(w)]
    matrix_B = [[random.randint(min_val, max_val) for _ in range(h) ] for _ in range(o)]

    with open(out_pth, 'w') as out_f:
        out_f.write("# Matrix A\n")
        out_f.write(f"{w} {o}\n")
        for i in matrix_A:
            for j in i:
                out_f.write(f"{j} ")
            out_f.write('\n')
        out_f.write('\n')

        out_f.write("# Matrix B\n")
        out_f.write(f"{o} {h}\n")
        for i in matrix_B:
            for j in i:
                out_f.write(f"{j} ")
            out_f.write('\n')
        out_f.write('\n')

        out_f.write("# WIDTH, HEIGHT, offset of tile\n")
        out_f.write(f"{t_w} {t_h} {o_k}")

def arg_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--WIDTH', type=int, required=True)
    parser.add_argument('--HEIGHT', type=int, required=True)
    parser.add_argument('--OFFSET_K', type=int, required=True)
    parser.add_argument('--OFFSET', type=int, required=True)
    parser.add_argument('--TILE_WIDTH', type=int, required=True)
    parser.add_argument('--TILE_HEIGHT', type=int, required=True)
    parser.add_argument('--MAX_VALUE', type=int, required=True)
    parser.add_argument('--MIN_VALUE', type=int, required=True)
    parser.add_argument('--OUT_PTH', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    random.seed(42)
    gen_matrix(
        args.WIDTH,
        args.HEIGHT,
        args.OFFSET,
        args.OFFSET_K,
        args.TILE_WIDTH,
        args.TILE_HEIGHT,
        args.MAX_VALUE,
        args.MIN_VALUE,
        args.OUT_PTH,
    )
    
    