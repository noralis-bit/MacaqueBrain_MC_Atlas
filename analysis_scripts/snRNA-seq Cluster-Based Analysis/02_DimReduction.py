# coding:utf-8
import os,re
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import warnings
warnings.simplefilter("ignore")

adata = sc.read_h5ad("nRNA_keep3Animal_1stQC.h5ad")
adata.obs["Date"] = adata.obs["Date"].astype("str")

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.raw = adata.copy()
### eliminate mito_genes with nuclear-encoded OXPHOS genes for state inference
sc.pp.highly_variable_genes(adata, n_top_genes = 3000, batch_key="orig.ident", subset=True) 
sc.pp.regress_out(adata, ["nCount_RNA","percent.mt"])

sc.tl.pca(adata)
sce.pp.bbknn(adata, batch_key='Date',use_rep = 'X_pca') 
sc.tl.umap(adata)
sc.tl.louvain(adata, resolution = 1)

adata.write_h5ad("nRNA_keep3Animal_1stQC_regMito_bbknn.h5ad")