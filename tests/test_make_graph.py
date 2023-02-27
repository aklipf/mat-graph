import unittest
import tempfile
import os
import urllib.request

import torch
from torch_geometric.loader import DataLoader

from tests.data import CSVDataset

import crystallographic_graph


class TestCrystallographicGraph(unittest.TestCase):
    batch = None

    def get_tmp_batch_(self):

        if TestCrystallographicGraph.batch is not None:
            return TestCrystallographicGraph.batch

        if "DATASET_PATH" in os.environ:
            os.makedirs(os.environ["DATASET_PATH"], exist_ok=True)

            if os.path.exists(os.path.join(os.environ["DATASET_PATH"], "test.csv")):
                local_filename = os.path.join(
                    os.environ["DATASET_PATH"], "test.csv")
            else:
                print("downloading to", os.path.join(
                    os.environ["DATASET_PATH"], "test.csv"))
                local_filename, _ = urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/test.csv",
                    os.path.join(os.environ["DATASET_PATH"], "test.csv"),
                )

            dataset = CSVDataset(local_filename)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                local_filename, _ = urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/txie-93/cdvae/main/data/carbon_24/test.csv",
                    os.path.join(tmp, "test.csv"),
                )

                dataset = CSVDataset(local_filename)

        dataloader = DataLoader(dataset, batch_size=len(dataset))

        for batch in dataloader:
            pass

        self.assertEqual(batch.num_atoms.shape[0], len(dataset))

        TestCrystallographicGraph.batch = batch

        return TestCrystallographicGraph.batch

    def test_make_graph_knn(self):
        batch = self.get_tmp_batch_()

        knn = 16

        edges = crystallographic_graph.make_graph(
            batch.cell, batch.pos, batch.num_atoms, knn=knn
        )

        self.assertEqual(edges.src.dtype, torch.long)
        self.assertEqual(edges.dst.dtype, torch.long)
        self.assertEqual(edges.cell.dtype, torch.long)
        self.assertEqual(edges.src.ndim, 1)
        self.assertEqual(edges.dst.ndim, 1)
        self.assertEqual(edges.cell.ndim, 2)
        self.assertEqual(edges.src.shape[0], edges.dst.shape[0])
        self.assertEqual(edges.dst.shape[0], edges.cell.shape[0])
        self.assertEqual(edges.cell.shape[1], 3)
        self.assertGreater(edges.src.shape[0], batch.pos.shape[0] * knn)

    def test_make_graph_cutoff(self):
        batch = self.get_tmp_batch_()

        cutoff = 5.0

        edges = crystallographic_graph.make_graph(
            batch.cell, batch.pos, batch.num_atoms, cutoff=cutoff
        )

        self.assertEqual(edges.src.dtype, torch.long)
        self.assertEqual(edges.dst.dtype, torch.long)
        self.assertEqual(edges.cell.dtype, torch.long)
        self.assertEqual(edges.src.ndim, 1)
        self.assertEqual(edges.dst.ndim, 1)
        self.assertEqual(edges.cell.ndim, 2)
        self.assertEqual(edges.src.shape[0], edges.dst.shape[0])
        self.assertEqual(edges.dst.shape[0], edges.cell.shape[0])
        self.assertEqual(edges.cell.shape[1], 3)

        e_ij = batch.pos[edges.dst] + edges.cell.float() - batch.pos[edges.src]
        v_ij = torch.bmm(
            batch.cell[batch.batch[edges.src]],
            e_ij.unsqueeze(2),
        ).squeeze(2)
        r_ij = v_ij.norm(dim=1)

        self.assertLessEqual(r_ij.max().item(), cutoff)

    def test_make_graph_mask(self):
        batch = self.get_tmp_batch_()

        cutoff = 5.0

        src_mask = torch.rand(batch.pos.shape[0]) > 0.5
        dst_mask = ~src_mask

        edges = crystallographic_graph.make_graph(
            batch.cell,
            batch.pos,
            batch.num_atoms,
            cutoff=cutoff,
            src_mask=src_mask,
            dst_mask=dst_mask,
        )

        edges_complet = crystallographic_graph.make_graph(
            batch.cell, batch.pos, batch.num_atoms, cutoff=cutoff
        )

        self.assertEqual(edges.src.dtype, torch.long)
        self.assertEqual(edges.dst.dtype, torch.long)
        self.assertEqual(edges.cell.dtype, torch.long)
        self.assertEqual(edges.src.ndim, 1)
        self.assertEqual(edges.dst.ndim, 1)
        self.assertEqual(edges.cell.ndim, 2)
        self.assertEqual(edges.src.shape[0], edges.dst.shape[0])
        self.assertEqual(edges.dst.shape[0], edges.cell.shape[0])
        self.assertEqual(edges.cell.shape[1], 3)

        self.assertTrue(src_mask[edges.src].all())
        self.assertTrue(dst_mask[edges.dst].all())

        is_src = src_mask[edges_complet.src]
        is_dst = dst_mask[edges_complet.dst]

        edges_complet.src = edges_complet.src[is_src & is_dst]
        edges_complet.dst = edges_complet.dst[is_src & is_dst]
        edges_complet.cell = edges_complet.cell[is_src & is_dst]

        self.assertEqual(edges_complet.src.shape[0], edges.src.shape[0])

        e = torch.cat(
            (edges.src.unsqueeze(1), edges.dst.unsqueeze(1), edges.cell), dim=1
        )

        for i in range(e.shape[1]):
            _, idx = torch.sort(e[:, i], stable=True)

            e = e[idx]
            edges.src = edges.src[idx]
            edges.dst = edges.dst[idx]
            edges.cell = edges.cell[idx]

        e_complet = torch.cat(
            (
                edges_complet.src.unsqueeze(1),
                edges_complet.dst.unsqueeze(1),
                edges_complet.cell,
            ),
            dim=1,
        )

        for i in range(e_complet.shape[1]):
            _, idx = torch.sort(e_complet[:, i], stable=True)

            e_complet = e_complet[idx]
            edges_complet.src = edges_complet.src[idx]
            edges_complet.dst = edges_complet.dst[idx]
            edges_complet.cell = edges_complet.cell[idx]

        self.assertTrue((edges_complet.src == edges.src).all())
        self.assertTrue((edges_complet.dst == edges.dst).all())
        self.assertTrue((edges_complet.cell == edges.cell).all())

    def test_make_graph_symetric(self):
        batch = self.get_tmp_batch_()

        knn = 16

        edges = crystallographic_graph.make_graph(
            batch.cell, batch.pos, batch.num_atoms, knn=knn, symetric=True, compute_reverse_idx=True
        )

        self.assertTrue(edges.reverse_idx is not None)
        self.assertTrue((edges.src == edges.dst[edges.reverse_idx]).all())
        self.assertTrue((edges.dst == edges.src[edges.reverse_idx]).all())
        self.assertTrue((edges.cell == -edges.cell[edges.reverse_idx]).all())
