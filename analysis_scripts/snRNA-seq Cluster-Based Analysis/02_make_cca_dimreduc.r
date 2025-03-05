pacman::p_load(Seurat,dplyr)
options(future.globals.maxSize = 1e+12)

seuObj <- readRDS("nRNA_keep3Animal_1stQC.Rds")

{
    seuObj[["RNA"]] <- split(seuObj[["RNA"]], seuObj = seuObj$Date)
    seuObj <- seuObj %>% SCTransform(vars.to.regress = c("nCount_RNA","percent.mt"),verbose = F) %>% RunPCA(verbose = F) %>% 
        IntegrateLayers(method = CCAIntegration, normalization.method = "SCT", verbose = F) %>% 
        FindNeighbors(dims = 1:30,reduction = "integrated.dr") %>% FindClusters(resolution = 1) %>% 
        RunUMAP(dims = 1:30,reduction = "integrated.dr", verbose = F)   
}


{
    seuObj %>% saveRDS("nRNA_keep3Animal_1stQC_regMito_cca.Rds")
    {pacman::p_load(reticulate); use_condaenv("Moics-R43", required = TRUE)}
    scCustomize::as.anndata(x = seuObj, assay = "SCT", file_path = ".",file_name = "nRNA_keep3Animal_1stQC_regMito_cca.h5ad", main_layer = "data", other_layers = NULL)
}
