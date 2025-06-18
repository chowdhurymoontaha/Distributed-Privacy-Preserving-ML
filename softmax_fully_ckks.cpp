#include "openfhe.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <memory>
#include <iomanip>
#include <cmath>

using namespace lbcrypto;

class BatchedPrivacyPreservingSoftmax {
private:
    CryptoContext<DCRTPoly> context;
    KeyPair<DCRTPoly> keyPair;
    
    uint32_t multDepth;
    uint32_t scaleModSize;
    uint32_t batchSize;
    
    // Cached encrypted constants (broadcasted across slots)
    std::shared_ptr<Ciphertext<DCRTPoly>> cached_ones;
    std::shared_ptr<Ciphertext<DCRTPoly>> cached_twos;
    std::shared_ptr<Ciphertext<DCRTPoly>> cached_halves;

public:
    BatchedPrivacyPreservingSoftmax(uint32_t depth = 25, uint32_t scale = 35, uint32_t batch = 1024) 
        : multDepth(depth), scaleModSize(scale), batchSize(batch) {
        setupCKKS();
        cacheBatchedConstants();
    }

    void setupCKKS() {
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetMultiplicativeDepth(multDepth);
        parameters.SetScalingModSize(scaleModSize);
        parameters.SetFirstModSize(60);
        parameters.SetBatchSize(batchSize);
        parameters.SetSecurityLevel(HEStd_128_classic);
        parameters.SetRingDim(1 << 17);  // 131072 for 128-bit security
        parameters.SetKeySwitchTechnique(HYBRID);

        context = GenCryptoContext(parameters);
        context->Enable(PKE);
        context->Enable(KEYSWITCH);
        context->Enable(LEVELEDSHE);

        keyPair = context->KeyGen();
        context->EvalMultKeyGen(keyPair.secretKey);
        
        // Generate rotation keys for SIMD operations
        std::vector<int32_t> rotations;
        for (int i = 1; i < static_cast<int>(batchSize); i <<= 1) {
            rotations.push_back(i); // +1,+2,+4,...
            rotations.push_back(-i);// -1,-2,-4,... 
        }
        // sequential sum for Average computation
        for (int i = 1; i <= static_cast<int>(batchSize/2); ++i) {
            rotations.push_back(i);// 1,2,3,4,...
        }
        context->EvalRotateKeyGen(keyPair.secretKey, rotations);
        
        std::cout << "Batched CKKS context initialized" << std::endl;
        std::cout << "Ring dimension: " << (1 << 17) << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Multiplicative depth: " << multDepth << std::endl;
    }

    void cacheBatchedConstants() {
        // Create vectors filled with constants for all slots
        std::vector<double> ones(batchSize, 1.0);
        std::vector<double> twos(batchSize, 2.0);
        std::vector<double> halves(batchSize, 0.5);
        
        auto ones_plain = context->MakeCKKSPackedPlaintext(ones);
        auto twos_plain = context->MakeCKKSPackedPlaintext(twos);
        auto halves_plain = context->MakeCKKSPackedPlaintext(halves);
        
        cached_ones = std::make_shared<Ciphertext<DCRTPoly>>(
            context->Encrypt(keyPair.publicKey, ones_plain));
        cached_twos = std::make_shared<Ciphertext<DCRTPoly>>(
            context->Encrypt(keyPair.publicKey, twos_plain));
        cached_halves = std::make_shared<Ciphertext<DCRTPoly>>(
            context->Encrypt(keyPair.publicKey, halves_plain));
            
        std::cout << "Batched constants cached across " << batchSize << " slots" << std::endl;
    }

    // Batched approximate exponential using SIMD
    Ciphertext<DCRTPoly> batchedApproximateExp(const Ciphertext<DCRTPoly>& x) {
        // Scale down input: x/8 to avoid overflow (applied to all slots)
        std::vector<double> scale_vec(batchSize, 0.125);
        auto scale_plain = context->MakeCKKSPackedPlaintext(scale_vec);
        auto scaled_x = context->EvalMult(x, scale_plain);
        
        // Taylor series: 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
        auto result = *cached_ones;  // All slots = 1.0
        auto x_power = scaled_x;
        
        // Precompute plaintext coefficients (broadcasted to all slots)
        std::vector<double> coefficients = {1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0};
        
        for (size_t i = 0; i < coefficients.size(); ++i) {
            std::vector<double> coeff_vec(batchSize, coefficients[i]);
            auto coeff_plain = context->MakeCKKSPackedPlaintext(coeff_vec);
            auto term = context->EvalMult(x_power, coeff_plain);
            result = context->EvalAdd(result, term);
            
            if (i < coefficients.size() - 1) {
                x_power = context->EvalMult(x_power, scaled_x);
            }
        }
        
        // Raise to 8th power efficiently: ((result²)²)²
        auto result_2 = context->EvalMult(result, result);
        auto result_4 = context->EvalMult(result_2, result_2);
        return context->EvalMult(result_4, result_4);
    }

    // Batched Newton-Raphson division (SIMD across all slots)
    Ciphertext<DCRTPoly> batchedApproximateDivision(const Ciphertext<DCRTPoly>& numerator, 
                                                   const Ciphertext<DCRTPoly>& denominator) {
        // Fixed initial guess for all slots
        std::vector<double> init_vec(batchSize, 0.1);
        auto init_plain = context->MakeCKKSPackedPlaintext(init_vec);
        auto x = context->Encrypt(keyPair.publicKey, init_plain);
        
        // Newton-Raphson iterations: x_{n+1} = x_n(2 - a*x_n): Iteratively converges to 1/b
        //a/b = a × (1/b)
        for (int i = 0; i < 6; ++i) {
            auto ax = context->EvalMult(denominator, x);
            auto two_minus_ax = context->EvalSub(*cached_twos, ax);
            x = context->EvalMult(x, two_minus_ax);
        }
        
        return context->EvalMult(numerator, x);
    }

    // SIMD sum reduction using rotations
    Ciphertext<DCRTPoly> batchedSumReduction(const Ciphertext<DCRTPoly>& packed_values, size_t num_elements) {
        auto result = packed_values;
        
        // Sum all elements using rotation-and-add
        for (size_t step = 1; step < num_elements; step <<= 1) {
            auto rotated = context->EvalRotate(result, step);
            result = context->EvalAdd(result, rotated);
        }
        
        return result;
    }

    // Compute average for stability (SIMD operation)
    Ciphertext<DCRTPoly> batchedComputeAverage(const Ciphertext<DCRTPoly>& packed_values, size_t num_elements) {
        auto sum = batchedSumReduction(packed_values, num_elements);
        
        // Divide by number of elements (broadcast to all slots)
        std::vector<double> size_inv_vec(batchSize, 1.0 / static_cast<double>(num_elements));
        auto size_inv_plain = context->MakeCKKSPackedPlaintext(size_inv_vec);
        return context->EvalMult(sum, size_inv_plain);
    }

    // Main batched softmax computation
    Ciphertext<DCRTPoly> batchedPrivateSoftmax(const Ciphertext<DCRTPoly>& encrypted_logits, size_t num_elements) {
        if (num_elements == 0 || num_elements > batchSize) {
            throw std::invalid_argument("Invalid number of elements for batched softmax");
        }

        std::cout << "Computing batched softmax on " << num_elements 
                  << " elements in single ciphertext..." << std::endl;

        // Calculate average for numerical stability
        auto stability_anchor = batchedComputeAverage(encrypted_logits, num_elements);

        // Subtract average from all logits (SIMD operation)
        auto shifted_logits = context->EvalSub(encrypted_logits, stability_anchor);

        // Compute exponentials (SIMD across all slots)
        auto exp_logits = batchedApproximateExp(shifted_logits);

        // Compute sum of exponentials using rotation-based reduction
        auto sum_exp = batchedSumReduction(exp_logits, num_elements);

        // Normalize: divide each exponential by sum (SIMD operation)
        auto softmax_result = batchedApproximateDivision(exp_logits, sum_exp);

        return softmax_result;
    }

    Ciphertext<DCRTPoly> encryptBatchedExternalData(const std::vector<double>& values) {
        if (values.size() > batchSize) {
            throw std::invalid_argument("Input size exceeds batch capacity");
        }
        
        std::cout << "Encrypting " << values.size() << " elements in single batched ciphertext..." << std::endl;
        
        // Pad with zeros if necessary
        std::vector<double> padded_values = values;
        padded_values.resize(batchSize, 0.0);
        
        auto plaintext = context->MakeCKKSPackedPlaintext(padded_values);
        auto encrypted_batch = context->Encrypt(keyPair.publicKey, plaintext);
        
        std::cout << "Data encrypted - using " << batchSize << " SIMD slots, " 
                  << values.size() << " active" << std::endl;
        return encrypted_batch;
    }

    // Decrypt batched results
    std::vector<double> authorizedBatchedDecrypt(const Ciphertext<DCRTPoly>& encrypted_batch, size_t num_elements) {
        std::cout << "Authorized batched decryption..." << std::endl;
        
        try {
            Plaintext result;
            context->Decrypt(keyPair.secretKey, encrypted_batch, &result);
            auto decoded = result->GetCKKSPackedValue();
            
            std::vector<double> results;
            results.reserve(num_elements);
            
            for (size_t i = 0; i < num_elements && i < decoded.size(); ++i) {
                results.push_back(decoded[i].real());
            }
            
            return results;
        } catch (const std::exception& e) {
            std::cout << "Batched decryption failed: " << e.what() << std::endl;
            return std::vector<double>(num_elements, 0.0);
        }
    }

    // Performance benchmark with batching optimization
    void batchedBenchmark(const std::vector<double>& external_logits) {
        std::cout << "\n=== OPTIMIZED Batched Privacy-Preserving Softmax Benchmark ===" << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Active elements: " << external_logits.size() << std::endl;
        std::cout << "SIMD efficiency: " << std::fixed << std::setprecision(1) 
                  << (100.0 * external_logits.size() / batchSize) << "%" << std::endl;
        
        // Encrypt all data in single batched ciphertext
        auto encrypted_batch = encryptBatchedExternalData(external_logits);

        auto start = std::chrono::high_resolution_clock::now();
        auto private_result = batchedPrivateSoftmax(encrypted_batch, external_logits.size());
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\n=== PERFORMANCE RESULTS ===" << std::endl;
        std::cout << "Batched computation time: " << duration.count() << " ms" << std::endl;
        std::cout << "Operations per ciphertext: " << external_logits.size() << std::endl;
        std::cout << "Effective throughput: " << std::fixed << std::setprecision(2)
                  << (external_logits.size() * 1000.0 / duration.count()) << " elements/sec" << std::endl;
        
        // Authorized decryption
        auto decrypted_results = authorizedBatchedDecrypt(private_result, external_logits.size());
        
        std::cout << "\n=== SOFTMAX RESULTS ===" << std::endl;
        std::cout << "Input logits:  [";
        for (size_t i = 0; i < external_logits.size(); ++i) {
            std::cout << std::fixed << std::setprecision(3) << external_logits[i];
            if (i < external_logits.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Softmax output: [";
        for (size_t i = 0; i < decrypted_results.size(); ++i) {
            std::cout << std::fixed << std::setprecision(6) << decrypted_results[i];
            if (i < decrypted_results.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Verify normalization
        double sum = 0.0;
        for (double val : decrypted_results) {
            sum += val;
        }
        std::cout << "Sum verification: " << std::fixed << std::setprecision(6) << sum 
                  << " (should be ~1.000000)" << std::endl;
        
    }
   
    PublicKey<DCRTPoly> getPublicKey() const {
        return keyPair.publicKey;
    }
    
    CryptoContext<DCRTPoly> getContext() const {
        return context;
    }
    
    uint32_t getBatchSize() const {
        return batchSize;
    }
};

class BatchedExternalDataProvider {
public:    
    static std::vector<double> generateSmallDataset() {
        return {1.2, 2.3, 0.8, 3.1, 1.7, 2.9, 0.5, 1.4, 3.0, 2.1, 1.8, 0.9};
    }
    
};


int main() {
    try {
        std::vector<uint32_t> batch_sizes = {16};
        
        for (uint32_t batch_size : batch_sizes) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "Testing with batch size: " << batch_size << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            
            BatchedPrivacyPreservingSoftmax batched_engine(25, 35, batch_size);
            
            // Testing with small dataset
            auto small_logits = BatchedExternalDataProvider::generateSmallDataset();
            batched_engine.batchedBenchmark(small_logits);           
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}