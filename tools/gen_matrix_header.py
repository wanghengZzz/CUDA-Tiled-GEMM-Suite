#!/usr/bin/env python3
import sys

def generate_header(input_file, output_file):
    with open(input_file) as f:
        # Remove empty lines and comments starting with '#'
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    def read_matrix(start):
        """Read a matrix starting at index 'start' in lines."""
        rows, cols = map(int, lines[start].split())
        values = []
        for i in range(rows):
            values += list(map(int, lines[start + 1 + i].split()))
        return rows, cols, values, start + 1 + rows

    # Parse matrices A and B
    A_m, A_n, A_vals, next_idx = read_matrix(0)
    B_m, B_n, B_vals, next_idx = read_matrix(next_idx)

    # Parse WIDTH, HEIGHT, offset
    WIDTH, HEIGHT, offset = map(int, lines[next_idx].split())

    # Write to header
    with open(output_file, "w") as out:
        out.write("// Auto-generated from matrix_data.txt\n")
        out.write("#pragma once\n")
        out.write("#ifdef INCLUDE_MATRIX_DATA\n\n")
        # Write matrix dimensions
        out.write("/* Compile-time constants */\n")
        out.write(f"constexpr const int A_m = {A_m}, A_n = {A_n};\n")
        out.write(f"constexpr const int B_m = {B_m}, B_n = {B_n};\n")
        out.write(f"constexpr const int WIDTH = {WIDTH}, HEIGHT = {HEIGHT}, offset = {offset};\n\n")
        out.write(f"constexpr const int m = {A_m}, n = {B_n};\n")

        out.write("/* Test matrices used for demonstration and validation */\n")

        # Write matrix A
        out.write(f"static const int A_host[{A_m}][{A_n}] = {{\n")
        for i in range(A_m):
            row = ", ".join(map(str, A_vals[i*A_n:(i+1)*A_n]))
            comma = "," if i != A_m - 1 else ""
            out.write(f"    {{{row}}}{comma}\n")
        out.write("};\n\n")

        # Write matrix B
        out.write(f"static const int B_host[{B_m}][{B_n}] = {{\n")
        for i in range(B_m):
            row = ", ".join(map(str, B_vals[i*B_n:(i+1)*B_n]))
            comma = "," if i != B_m - 1 else ""
            out.write(f"    {{{row}}}{comma}\n")
        out.write("};\n")
        out.write("#endif\n")

    print(f"✅ Generated '{output_file}' with A={A_m}x{A_n}, B={B_m}x{B_n}, TILE=({HEIGHT}x{WIDTH}), offset={offset}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_matrix_header.py data/matrix_data.txt include/matrix_data.h")
        sys.exit(1)
    generate_header(sys.argv[1], sys.argv[2])
