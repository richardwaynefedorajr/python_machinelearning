blocks = [1, 3, 2]
can_stack = True
top_block = 0

while len(blocks) > 0:
    next_block = blocks.pop(-1) if blocks[-1] > blocks[0] else blocks.pop(0)
    if top_block == 0:
        top_block = next_block
    elif next_block <= top_block:
        top_block = next_block
    else:
        can_stack = False
        
print('Yes') if can_stack else print('No')
