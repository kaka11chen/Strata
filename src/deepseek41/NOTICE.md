# Source contract

The metadata names, latest-source KV/index interpretation and mixed-expert tensor names
follow `tools/v41/engine/ds41_geometry.hpp` and `tools/v41/v41_gguf_meta.py` in
[Xolotl at b3452666e07a6374d06bb8c2185dffa271e8f65f](https://github.com/Indras-Mirror/Xolotl/tree/b3452666e07a6374d06bb8c2185dffa271e8f65f/tools/v41).
The small artifact adapter here is adapted to Strata's existing GGUF reader; it does not
import Xolotl's execution engine or its routing bias policy.

Xolotl is MIT licensed. Its notice is retained below.

MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
