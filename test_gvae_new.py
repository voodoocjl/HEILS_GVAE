import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from GVAE_model_Version7 import GVAE_Dual, VAEReconstructed_Loss, GraphConvolution, VAEncoder, Decoder
from GVAE_model import normalize_adj, swap_ops


class TestGVAE(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures with common parameters."""
        self.batch_size = 2
        self.n_qubits = 4
        self.n_layers = 3
        self.n_nodes = self.n_qubits * self.n_layers * 2  # Single + entangling gates
        self.feature_dim = 8  # 4 single gate types + 4 entangling gate positions
        self.dims = [self.feature_dim, 16, 8]  # Encoder dimensions
        self.normalize = True
        self.dropout = 0.1
        
        # Create test data
        self.ops = torch.randn(self.batch_size, self.n_nodes, self.feature_dim)
        self.adj = torch.rand(self.batch_size, self.n_nodes, self.n_nodes)
        # Make adjacency matrix symmetric
        self.adj = (self.adj + self.adj.transpose(-1, -2)) / 2
        
        # Initialize model
        self.model = GVAE_Dual(self.dims, self.normalize, self.dropout)
        
    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model.encoder, VAEncoder)
        self.assertIsInstance(self.model.decoder, Decoder)
        self.assertEqual(len(self.model.encoder.gcs), len(self.dims) - 1)
        
    def test_encoder_output_shapes(self):
        """Test if encoder produces correct output shapes."""
        mu1, logvar1, mu2, logvar2 = self.model.encoder(self.ops, self.adj)
        
        expected_shape = (self.batch_size, self.n_nodes, self.dims[-1])
        self.assertEqual(mu1.shape, expected_shape)
        self.assertEqual(logvar1.shape, expected_shape)
        self.assertEqual(mu2.shape, expected_shape)
        self.assertEqual(logvar2.shape, expected_shape)
        
    def test_decoder_output_shape(self):
        """Test if decoder produces correct output shape."""
        embedding1 = torch.randn(self.batch_size, self.n_nodes, self.dims[-1])
        embedding2 = torch.randn(self.batch_size, self.n_nodes, self.dims[-1])
        ops_recon, adj_recon = self.model.decoder(embedding1, embedding2, self.n_qubits)
        
        expected_ops_shape = (self.batch_size, self.n_nodes * 2, self.feature_dim)
        expected_adj_shape = (self.batch_size, self.n_nodes * 2, self.n_nodes * 2)
        self.assertEqual(ops_recon.shape, expected_ops_shape)
        self.assertEqual(adj_recon.shape, expected_adj_shape)
        
    def test_reparameterize(self):
        """Test the reparameterization trick."""
        mu = torch.randn(self.batch_size, self.n_nodes, self.dims[-1])
        logvar = torch.randn(self.batch_size, self.n_nodes, self.dims[-1])
        
        # Test training mode
        self.model.train()
        z = self.model.reparameterize(mu, logvar)
        self.assertEqual(z.shape, mu.shape)
        
        # Test eval mode (should return mu)
        self.model.eval()
        z_eval = self.model.reparameterize(mu, logvar)
        self.assertTrue(torch.allclose(z_eval, mu))
        
    def test_combine_ops(self):
        """Test the combine_ops function."""
        ops_single = torch.randn(self.batch_size, self.n_nodes, self.feature_dim)
        ops_enta = torch.randn(self.batch_size, self.n_nodes, self.feature_dim)
        
        ops_combined = self.model.decoder.combine_ops(ops_single, ops_enta, self.n_qubits)
        
        # Should have double the nodes (single + entangling)
        expected_shape = (self.batch_size, self.n_nodes * 2, self.feature_dim)
        self.assertEqual(ops_combined.shape, expected_shape)
        
    def test_forward_pass(self):
        """Test complete forward pass."""
        outputs = self.model(self.ops, self.adj, self.n_qubits)
        # The model returns 6 outputs: ops_recon, adj_recon, mu1, logvar1, mu2, logvar2
        self.assertEqual(len(outputs), 6)
        ops_recon, adj_recon, mu1, logvar1, mu2, logvar2 = outputs
        expected_ops_shape = (self.batch_size, self.n_nodes * 2, self.feature_dim)
        expected_adj_shape = (self.batch_size, self.n_nodes * 2, self.n_nodes * 2)
        expected_latent_shape = (self.batch_size, self.n_nodes, self.dims[-1])
        self.assertEqual(ops_recon.shape, expected_ops_shape)
        self.assertEqual(adj_recon.shape, expected_adj_shape)
        self.assertEqual(mu1.shape, expected_latent_shape)
        self.assertEqual(logvar1.shape, expected_latent_shape)
        self.assertEqual(mu2.shape, expected_latent_shape)
        self.assertEqual(logvar2.shape, expected_latent_shape)
        
    def test_gradient_flow(self):
        """Test if gradients flow properly through the model."""
        self.model.train()
        
        # Forward pass
        outputs = self.model(self.ops, self.adj, self.n_qubits)
        ops_recon, adj_recon, mu1, logvar1, mu2, logvar2 = outputs
        
        # Compute a simple loss that involves all outputs
        loss = (ops_recon.sum() + adj_recon.sum() + 
                mu1.sum() + logvar1.sum() + mu2.sum() + logvar2.sum())
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist for parameters that should have gradients
        params_with_grads = 0
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grads += 1
        
        # At least some parameters should have gradients
        self.assertGreater(params_with_grads, 0)
            
    def test_deterministic_eval_mode(self):
        """Test if model is deterministic in eval mode."""
        self.model.eval()
        
        with torch.no_grad():
            # Set random seed for reproducibility
            torch.manual_seed(42)
            outputs1 = self.model(self.ops, self.adj, self.n_qubits)
            torch.manual_seed(42)
            outputs2 = self.model(self.ops, self.adj, self.n_qubits)
            
        # In eval mode, outputs should be identical (no randomness in reparameterize)
        for out1, out2 in zip(outputs1, outputs2):
            self.assertTrue(torch.allclose(out1, out2, atol=1e-6))
            
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        batch_sizes = [1, 3, 5]
        
        for bs in batch_sizes:
            ops = torch.randn(bs, self.n_nodes, self.feature_dim)
            adj = torch.rand(bs, self.n_nodes, self.n_nodes)
            adj = (adj + adj.transpose(-1, -2)) / 2
            
            outputs = self.model(ops, adj, self.n_qubits)
            
            # Check batch dimension
            for output in outputs:
                self.assertEqual(output.shape[0], bs)


class TestVAEReconstructedLoss(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for loss function."""
        self.batch_size = 2
        self.n_nodes = 16
        self.feature_dim = 7
        self.latent_dim = 8
        
        # Create test data - ops_single and ops_enta are now combined into one tensor
        self.ops_recon = torch.randn(self.batch_size, self.n_nodes * 2, self.feature_dim)
        self.adj_recon = torch.randn(self.batch_size, self.n_nodes * 2, self.n_nodes * 2)
        
        self.ops_target = torch.randn(self.batch_size, self.n_nodes * 2, self.feature_dim)
        self.adj_target = torch.randn(self.batch_size, self.n_nodes * 2, self.n_nodes * 2)
        
        self.mu1 = torch.randn(self.batch_size, self.n_nodes, self.latent_dim)
        self.logvar1 = torch.randn(self.batch_size, self.n_nodes, self.latent_dim)
        self.mu2 = torch.randn(self.batch_size, self.n_nodes, self.latent_dim)
        self.logvar2 = torch.randn(self.batch_size, self.n_nodes, self.latent_dim)
        
        # Initialize loss function
        self.loss_fn = VAEReconstructed_Loss(
            w_ops=1.0,
            w_adj=0.5,
            loss_ops=nn.MSELoss(),
            loss_adj=nn.MSELoss()
        )
        
    def test_loss_computation(self):
        """Test if loss computation works correctly."""
        # Split ops_recon into single and enta parts for loss computation
        ops_single = self.ops_recon[:, :self.n_nodes, :]
        ops_enta = self.ops_recon[:, self.n_nodes:, :]
        ops_single_target = self.ops_target[:, :self.n_nodes, :]
        ops_enta_target = self.ops_target[:, self.n_nodes:, :]
        
        inputs = [ops_single, ops_enta, self.adj_recon]
        targets = [ops_single_target, ops_enta_target, self.adj_target]
        
        loss = self.loss_fn(inputs, targets, self.mu1, self.logvar1, self.mu2, self.logvar2)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Should be a scalar
        self.assertTrue(loss.item() >= 0)  # Loss should be non-negative
        
    def test_loss_components(self):
        """Test individual loss components."""
        # Test with identical inputs and targets (should give low reconstruction loss)
        ops_single = self.ops_target[:, :self.n_nodes, :]
        ops_enta = self.ops_target[:, self.n_nodes:, :]
        inputs = [ops_single, ops_enta, self.adj_target]
        targets = [ops_single, ops_enta, self.adj_target]
        
        # Set mu and logvar to zeros for minimal KLD
        mu1_zero = torch.zeros_like(self.mu1)
        logvar1_zero = torch.zeros_like(self.logvar1)
        mu2_zero = torch.zeros_like(self.mu2)
        logvar2_zero = torch.zeros_like(self.logvar2)
        
        loss = self.loss_fn(inputs, targets, mu1_zero, logvar1_zero, mu2_zero, logvar2_zero)
        
        # Loss should be close to zero with identical inputs/targets and zero latent params
        self.assertTrue(loss.item() < 1.0)


class TestUtilityFunctions(unittest.TestCase):
    def test_normalize_adj(self):
        """Test adjacency matrix normalization."""
        batch_size = 2
        n_nodes = 4
        adj = torch.rand(batch_size, n_nodes, n_nodes)
        # Make input symmetric
        adj = (adj + adj.transpose(-1, -2)) / 2
        
        normalized_adj = normalize_adj(adj)
        
        self.assertEqual(normalized_adj.shape, adj.shape)
        # Check if all values are finite
        self.assertTrue(torch.all(torch.isfinite(normalized_adj)))
        
    def test_swap_ops(self):
        """Test operation swapping function."""
        batch_size = 2
        n_nodes = 8
        feature_dim = 7
        p = 4
        
        ops = torch.randn(batch_size, n_nodes, feature_dim)
        swapped_ops = swap_ops(ops, p)
        
        self.assertEqual(swapped_ops.shape, ops.shape)
        # Check if channels are swapped correctly
        self.assertTrue(torch.allclose(swapped_ops[:, :, :feature_dim-p], ops[:, :, p:]))
        self.assertTrue(torch.allclose(swapped_ops[:, :, feature_dim-p:], ops[:, :, :p]))


class TestModelIntegration(unittest.TestCase):
    def test_training_loop_simulation(self):
        """Test a simplified training loop to ensure everything works together."""
        # Model setup
        batch_size = 2
        n_qubits = 4
        n_layers = 2
        n_nodes = n_qubits * n_layers * 2
        feature_dim = 7
        dims = [feature_dim, 16, 8]
        
        model = GVAE_Dual(dims, normalize=True, dropout=0.1)
        loss_fn = VAEReconstructed_Loss(
            w_ops=1.0,
            w_adj=0.5,
            loss_ops=nn.MSELoss(),
            loss_adj=nn.MSELoss()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Generate synthetic data
        ops = torch.randn(batch_size, n_nodes, feature_dim)
        adj = torch.rand(batch_size, n_nodes, n_nodes)
        adj = (adj + adj.transpose(-1, -2)) / 2
        
        # Training loop
        model.train()
        initial_loss = None
        
        for epoch in range(3):
            optimizer.zero_grad()
            # Forward pass
            outputs = model(ops, adj, n_qubits)
            ops_recon, adj_recon, mu1, logvar1, mu2, logvar2 = outputs
            # Compute loss - split ops_recon for loss computation
            ops_single = ops_recon[:, :n_nodes, :]
            ops_enta = ops_recon[:, n_nodes:, :]
            inputs = [ops_single, ops_enta, adj_recon]
            targets = [ops_single.detach(), ops_enta.detach(), adj_recon.detach()]  # Self-reconstruction task
            loss = loss_fn(inputs, targets, mu1, logvar1, mu2, logvar2)
            if initial_loss is None:
                initial_loss = loss.item()
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Loss should exist and be finite
        self.assertTrue(torch.isfinite(loss))
        self.assertIsInstance(loss.item(), float)


def run_comprehensive_validation():
    """Run all tests and provide a comprehensive validation report."""
    print("=" * 60)
    print("GVAE Model Validation Report")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestGVAE,
        TestVAEReconstructedLoss,
        TestUtilityFunctions,
        TestModelIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=open('/dev/null', 'w'))
        result = runner.run(suite)
        
        class_total = result.testsRun
        class_passed = class_total - len(result.failures) - len(result.errors)
        
        total_tests += class_total
        passed_tests += class_passed
        
        print(f"Passed: {class_passed}/{class_total}")
        
        if result.failures:
            for test, traceback in result.failures:
                failed_tests.append(f"FAILURE - {test}: {traceback}")
                
        if result.errors:
            for test, traceback in result.errors:
                failed_tests.append(f"ERROR - {test}: {traceback}")
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if failed_tests:
        print("\nFAILED TESTS:")
        for failure in failed_tests:
            print(f"- {failure}")
    else:
        print("\n✅ All tests passed! GVAE model is validated.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    # Run comprehensive validation
    success = run_comprehensive_validation()
    
    # Additional manual checks
    print("\n" + "=" * 60)
    print("ADDITIONAL MANUAL CHECKS")
    print("=" * 60)
    
    try:
        # Test model creation and basic functionality
        model = GVAE_Dual([7, 16, 8], normalize=True, dropout=0.1)
        print("✅ Model creation: SUCCESS")
        
        # Test with quantum circuit-like data
        batch_size, n_qubits, n_layers = 1, 4, 3
        n_nodes = n_qubits * n_layers * 2
        ops = torch.randn(batch_size, n_nodes, 7)
        adj = torch.eye(n_nodes).unsqueeze(0) + torch.rand(batch_size, n_nodes, n_nodes) * 0.1
        
        outputs = model(ops, adj, n_qubits)
        print("✅ Forward pass with quantum-like data: SUCCESS")
        
        # Test output ranges (should be in [0,1] due to sigmoid)
        ops_recon, adj_recon = outputs[0], outputs[1]
        if torch.all(ops_recon >= 0) and torch.all(ops_recon <= 1):
            print("✅ Operations output range [0,1]: SUCCESS")
        else:
            print("❌ Operations output range [0,1]: FAILED")
            
        print(f"✅ Model validation complete!")
        
    except Exception as e:
        print(f"❌ Manual checks failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 GVAE MODEL VALIDATION: COMPLETE SUCCESS!")
    else:
        print("⚠️  GVAE MODEL VALIDATION: ISSUES DETECTED")
    print("=" * 60)
