# Why .contiguous() Exists

Oct 23, 2026 · @Yash

A tensor's shape is how PyTorch shows it to you; its memory is one flat line of numbers. `.contiguous()` is needed when those two orders stop matching, and `view` refuses to guess.

## Memory is one flat line

A tensor can look 2D, but memory has no rows or columns. It stores the numbers one after another.

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])     # shape (2, 3)
```

In memory this is `1 2 3 4 5 6`. To know where element `[i][j]` lives, PyTorch keeps **strides**: how many slots to jump to move one step along each dimension.

```python
t.stride()   # (3, 1)
```

Next row = jump 3 slots. Next column = jump 1 slot. So `t[1][2]` sits at slot 1 × 3 + 2 × 1 = 5, which holds the value 6.

## Transpose moves no data

Transpose is fast because it copies nothing. It only swaps the strides.

```python
tt = t.transpose(0, 1)   # shape (3, 2)
# looks like:
# [[1, 4],
#  [2, 5],
#  [3, 6]]
tt.stride()          # (1, 3)
tt.is_contiguous()   # False
```

Memory still holds `1 2 3 4 5 6`. PyTorch just walks it in a different pattern. Now the two orders disagree:

|  | Order |
| --- | --- |
| What you see, row by row | 1 4 2 5 3 6 |
| What memory holds | 1 2 3 4 5 6 |

A tensor is **contiguous** when reading it row by row walks straight through memory, slot 0, slot 1, slot 2, with no jumping. `t` is contiguous. `tt` is not.

## Why view breaks

`view` gives the same memory a new shape. It never moves data, so it reads memory in memory order.

```python
tt.view(6)   # RuntimeError: view size is not compatible with input tensor's size and stride
```

You'd expect `[1, 4, 2, 5, 3, 6]`. But reading memory in order gives `1 2 3 4 5 6`, and no single stride can produce the order you want. Rather than hand you the wrong numbers, `view` refuses.

`.contiguous()` fixes it by making a real copy, laid out in the order you see:

```python
tt.contiguous().view(6)   # tensor([1, 4, 2, 5, 3, 6])
```

| Method | Copies data? | Works on non-contiguous tensors? |
| --- | --- | --- |
| `view` | Never | No, raises an error |
| `contiguous` | Only if the tensor isn't contiguous already | Yes, that's its job |
| `reshape` | Only when it has to | Yes, it views when it can and copies when it must |

`x.contiguous().view(...)` and `x.reshape(...)` give the same result. Writing out `contiguous` makes the copy visible in your code.

## Where it shows up: multi-head attention

In multi-head attention, each head's output has shape `(B, H, T, d)`: batch, heads, tokens, features per head. To merge the heads back into one vector per token, you transpose and then view:

```python
out = out.transpose(1, 2).contiguous().view(B, T, H * d)
```

After `transpose(1, 2)` the shape reads `(B, T, H, d)`, but memory is still in `(B, H, T, d)` order: all of head 0's tokens, then all of head 1's tokens.

Merging `H` and `d` into one dimension needs each token's head outputs side by side in memory: token 0 head 0, token 0 head 1, and so on. They aren't, so `.contiguous()` copies them into that order first. Then `view` can glue each token's heads together, which is the same as concatenating the heads.

## Try it

The same attention code splits heads with a view and no `.contiguous()`:

```python
q = self.query(x).view(B, T, H, d)   # (B, T, head_size) -> (B, T, H, d)
```

Run `.is_contiguous()` on `self.query(x)`, then work out why splitting the last dimension into `(H, d)` never needs a copy. Hint: write out the memory order of one token's `head_size` features, then the order `(H, d)` reads them in.
