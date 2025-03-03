pacman::p_load(Seurat,dplyr,ggplot2);options(future.globals.maxSize = 1e+12)
Trim_IQR <- function(values,th = 1.5 ) {
    {
        Q1 <- quantile(values, 0.25, na.rm = TRUE)
        Q3 <- quantile(values, 0.75, na.rm = TRUE) 
        IQR <- Q3 - Q1
        lower_ <- Q1 - th * IQR
        upper_ <- Q3 + th * IQR
    }
    return(values > lower_ & values < upper_)
}


# Generate Quality Control Metrics and Merge Libraries ------------------------------------------------------------
mito_genes = c("ND6","COX3","COX1","ND5","ND4","ND2","ND4L","ATP8","CYTB","COX2","ND3","ATP6","ND1")
sample_Info <- readxl::read_xlsx("motor_cortex_libinfo.xlsx", sheet = 'rna.matrix')[,1:9] %>% filter(!is.na(Matrix_Directory)) %>% dplyr::rename("orig.ident" = "LibIndex") %>% data.frame()
lapply(1:nrow(sample_Info),function(i){
    counts = Read10X(sample_Info[i, "Matrix_Directory"], gene.column = 1)
    rownames(counts) <- gsub("MT_","",rownames(counts))
    obj <- CreateSeuratObject(counts = counts, project = sample_Info[i, "orig.ident"])
    {### doublets detection
        .sce <- Seurat::as.SingleCellExperiment(obj) %>% suppressWarnings
        .tbl <- scDblFinder::scDblFinder(.sce,returnType = "table",verbose = FALSE)[colnames(obj),c("score","class")] %>% setNames(c('doublet_score','doublet_class'))
        obj@meta.data <- obj@meta.data %>% cbind(.tbl)
    }
    obj[["percent.mt"]] <- PercentageFeatureSet(obj, features  = mito_genes)
    obj$Exprlevel <- round(obj$nCount_RNA / obj$nFeature_RNA, 2) 
    obj@meta.data <- obj@meta.data %>% tibble::rownames_to_column("celln") %>% plyr::join(sample_Info[i,],by = "orig.ident") %>% tibble::column_to_rownames("celln")
    
    return({obj})
}) %>% (\(.) merge(.[[1]],.[-1]))() -> combine_obj

combine_obj %>% qs::qsave(file = "combine_obj_raw.qs")
combine_obj@meta.data %>% write.table("combine_obj_raw_meta.tsv",sep = "\t",row.names = T, quote = F)

# Data Filtering ------------------------------------------------------------
{
    metaDT <- read.delim("combine_obj_raw_meta.tsv")
    metaDT_Final <- metaDT %>% filter(
        percent.mt < 5,
        doublet_class == "singlet",
        Animal != "MQC277"
    )
    
    Valid <- Trim_IQR(metaDT_Final$nFeature_RNA, th = 1.5) &
        Trim_IQR(metaDT_Final$nCount_RNA, th = 1.5) &
        Trim_IQR(metaDT_Final$Exprlevel, th = 1.5)
    
    Valid_CB <- metaDT_Final[Valid,] %>% filter(nFeature_RNA > 500,nCount_RNA > 1000,Exprlevel > 1.25) %>% rownames()
}

seuObj_raw <- qs::qread("combine_obj_raw.qs")
seuObj_new <- seuObj_raw %>% subset(cells = Valid_CB)

{
    seuObj_new %>% saveRDS("nRNA_keep3Animal_1stQC.Rds")
    {pacman::p_load(reticulate); use_condaenv("Moics-R43", required = TRUE)}
    scCustomize::as.anndata(x = seuObj_new,assay = "RNA", file_path = ".",file_name = "nRNA_keep3Animal_1stQC.h5ad", main_layer = "counts", other_layers = NULL)
}
