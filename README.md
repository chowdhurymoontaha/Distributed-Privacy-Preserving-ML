# Distributed-Privacy-Preserving-ML
OpenFHE, C++
This implementation provides an encrypted softmax computation that processes multiple elements simultaneously using CKKS homomorphic encryption. The softmax function is computed entirely on encrypted data without ever decrypting the inputs, ensuring complete privacy preservation.

# Key Features
SIMD Batch Processing: processes multiple elements in parallel using OpenFHE's batching capabilities
Optimized Performance: rotation-based reductions and cached constants
Numerical Stability: mean subtraction for stable softmax computation
Configurable Parameters: adjustable batch sizes, multiplicative depth, and security levels

# Mathematical Operations
Approximate Exponential: Taylor series expansion with input scaling
Newton-Raphson Division: Iterative method for encrypted division
Rotation-based Reduction: Efficient sum computation using SIMD rotations
Numerical Stabilization: Mean subtraction to prevent overflow
