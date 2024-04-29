import argparse
import textwrap

parser = argparse.ArgumentParser(prog="prova",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        description = textwrap.dedent("""
        """))

parser.add_argument("-a", "--aaa", type = str, default="aa")
parser.add_argument("-b", "--bbb", type = str)
parser.add_argument("-c", "--ccc", nargs="?", const = "cc",type = str)


args = parser.parse_args()
print(vars(args).values())

def abc(a,b):
    return a + b

possible_arguments = [""]
command_shell = f"python travel.py"
for argument in vars(args).keys():
    value = str(vars(args)[argument])
    if (value != "None") and (argument in possible_arguments):
        command_shell = command_shell + f" -{argument} {value}"

print(command_shell)