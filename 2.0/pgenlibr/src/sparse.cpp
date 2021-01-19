#include "pgenlibr.h"
#include <chrono>
using namespace std; 
using namespace std::chrono;

static constexpr int ROW_BLOCK = 8;
static constexpr int COL_BLOCK = 16;

void RPgenReader::LoadSparse(sparse_snp &x, const int *variant_subset, const uint32_t vsubset_size)
{
    if (!_info_ptr)
    {
        stop("pgen is closed");
    }
    x.no = _subset_size;
    x.ni = vsubset_size;

    x.blk_ptr = (uint64_t*)malloc(sizeof(uint64_t) * (x.rowblock * x.colblock + 1 ));
    x.xim = (double*)malloc(sizeof(double)*vsubset_size);
    for(uint32_t i = 0; i <  x.rowblock * x.colblock + 1; ++i){
        x.blk_ptr[i] = 0;
    }

    uint32_t row_block_size = _subset_size/ x.rowblock;
    uint32_t col_block_size = vsubset_size / x.colblock;

    // could tune this parameter for best performance
    uint32_t max_simple_difflist_len = _subset_size / 32;
    uint32_t difflist_len;
    uint32_t difflist_common_geno;

    const uint32_t difflist_len_upper_bound = 2 * (_info_ptr->raw_sample_ct / plink2::kPglMaxDifflistLenDivisor);


    // allocate memory to load the genovec
    uintptr_t *main_raregeno_0;
    if (plink2::cachealigned_malloc(
            plink2::kBytesPerVec + ((difflist_len_upper_bound + 3)/4), &main_raregeno_0)) {
        stop("Out of memory");
    }
    uintptr_t *main_raregeno = main_raregeno_0 + 2;

    uint32_t *difflist_sample_ids = (uint32_t *)malloc(sizeof(uint32_t) * difflist_len_upper_bound);
    uint32_t total_genovec = 0;

    // First pass
    for (uint32_t i = 0; i < vsubset_size; ++i)
    {
        plink2::PglErr reterr = PgrGetDifflistOrGenovec(_subset_include_vec, _subset_index, _subset_size, max_simple_difflist_len, variant_subset[i] - 1, _state_ptr, _pgv.genovec, &difflist_common_geno, main_raregeno, difflist_sample_ids, &difflist_len);
        if (reterr != plink2::kPglRetSuccess)
        {
            char errstr_buf[256];
            sprintf(errstr_buf, "PgrGetD() error %d", static_cast<int>(reterr));
            stop(errstr_buf);
        }
        STD_ARRAY_DECL(uint32_t, 4, genocounts); // impute mean

        if(difflist_common_geno == UINT32_MAX){
            plink2::ZeroTrailingNyps(_subset_size, _pgv.genovec);
            plink2::GenoarrCountFreqsUnsafe(_pgv.genovec, _subset_size, genocounts);
            const double numer = plink2::u63tod(genocounts[1] + 2 * genocounts[2]);
            const double denom = plink2::u31tod(_subset_size - genocounts[3]);
            x.xim[i] = (numer/denom);
            total_genovec++;
            continue;
        }

        // It's possible, though unlikely, that in a subset the difflist_common_geno is 1 or 2.
        // This case is currently not handled
        if(difflist_common_geno != 0){
            Rprintf("difflist common geno is %d\n", difflist_common_geno);
            stop("The common geno for sparse columns must be 0\n");
        }


        plink2::ZeroTrailingNyps(difflist_len, main_raregeno);
        plink2::GenoarrCountFreqsUnsafe(main_raregeno, difflist_len, genocounts);
        const double numer = plink2::u63tod(genocounts[1] + 2 * genocounts[2]);
        const double denom = plink2::u31tod(_subset_size - genocounts[3]);
        x.xim[i] = (numer / denom);

        // Otherwise we got a difflist
        uint32_t col_block_ind = i / col_block_size;
        if(col_block_ind == x.colblock){
            col_block_ind--;
        }

        uint32_t block_ind_start = 1 + col_block_ind * x.rowblock;
        for(uint32_t j = 0; j < difflist_len; ++j){
            uint32_t row_block_ind = difflist_sample_ids[j]/row_block_size;
            if(row_block_ind == x.rowblock){
                row_block_ind--;
            }
            x.blk_ptr[block_ind_start + row_block_ind]++;
        }
    }

    for(int i = 0; i < x.rowblock * x.colblock + 1; ++i){
        std::cout << x.blk_ptr[i] << std::endl;
    }
    std::cout << "total genovec is " << total_genovec << std::endl;
    x.dense_ind = (uint32_t*)malloc(sizeof(uint32_t) * total_genovec);
    x.ndense = total_genovec;
    // we want each column of genovec to be cacheline aligned
    // first term below is the number of cacheline needed for each column
    size_t genovec_cacheline_ct = plink2::DivUp(_subset_size, plink2::kCacheline * 4);
    x.genovec_word_ct = genovec_cacheline_ct * plink2::kWordsPerCacheline;
    if (plink2::cachealigned_malloc(
            genovec_cacheline_ct * plink2::kCacheline * total_genovec, &x.genovec)) {
        stop("Out of memory");
    }
    uint32_t genovec_iter = 0;

    // Now x.blk_ptr stores the nnz of each block, we can allocate the main data of this sparse matrix
    uint64_t max_block_size = 0;
    for(uint32_t i = 1; i <  x.rowblock * x.colblock + 1; ++i){
        uint64_t current_size = x.blk_ptr[i];
        // want cacheline aligned diffvec
        uint64_t current_size_with_padding = plink2::DivUp(current_size, plink2::kNypsPerCacheline) * plink2::kNypsPerCacheline;
        if(current_size_with_padding > max_block_size){
            max_block_size = current_size_with_padding;
        }
        x.blk_ptr[i] = current_size_with_padding + x.blk_ptr[i - 1];
    }

    for(uint32_t i = 0; i <  x.rowblock * x.colblock + 1; ++i){
        std::cout << x.blk_ptr[i] << std::endl;
    }
    uint64_t total_nnz = x.blk_ptr[x.rowblock * x.colblock];
    std::cout << "max_block_size " << max_block_size << std::endl;
    std::cout << "total nnz is " << total_nnz << std::endl;

    // Now malloc the difflist vec, indices
    if (plink2::cachealigned_malloc(
            sizeof(CSB_ind)*total_nnz, &x.index)) {
        stop("Out of memory");
    }

    // total_nnz is guaranteed to be a multiple of 256
    if (plink2::cachealigned_malloc(
            total_nnz/4, &x.diffvec)) {
        stop("Out of memory");
    }

    // allocate a mask for copy, very conservative estimate of size
    uint32_t mask_size_est = plink2::DivUp(x.no, x.rowblock * plink2::kBitsPerWordD2); 
    uintptr_t *mask = (uintptr_t*)malloc(sizeof(uintptr_t) * mask_size_est);
    for(uint32_t i = 0; i < mask_size_est; ++i){
        mask[i] = ~static_cast<uintptr_t>(0);
    }

    // Second pass, write the sparse block snp matrix
    uint64_t * row_block_nnz_count = (uint64_t *)malloc(sizeof(uint64_t)*x.rowblock);
    for(uint32_t j = 0; j < x.colblock; ++j){
        uint32_t start = j * col_block_size;
        uint32_t end = (j + 1) * col_block_size;
        if (j == x.colblock - 1) {
            end = vsubset_size;
        }
        for (uint32_t i = 0; i < x.rowblock; ++i) {
            row_block_nnz_count[i] = 0;
        }

        for (uint32_t colind = start; colind < end; ++colind)
        {
            plink2::PglErr reterr = PgrGetDifflistOrGenovec(_subset_include_vec, _subset_index, _subset_size, max_simple_difflist_len, variant_subset[colind] - 1, _state_ptr, _pgv.genovec, &difflist_common_geno, main_raregeno, difflist_sample_ids, &difflist_len);
            if (reterr != plink2::kPglRetSuccess) {
                char errstr_buf[256];
                sprintf(errstr_buf, "PgrGetD() error %d", static_cast<int>(reterr));
                stop(errstr_buf);
            }

            if (difflist_common_geno == UINT32_MAX) {
                uintptr_t *col_ptr = &(x.genovec[genovec_iter * x.genovec_word_ct]);
                memcpy(col_ptr, _pgv.genovec, genovec_cacheline_ct * plink2::kCacheline);
                plink2::ZeroTrailingNyps(_subset_size, col_ptr);
                x.dense_ind[genovec_iter] = colind;
                genovec_iter++;
                continue;
            }

            uint32_t prev_row_block_ind = 0;
            uint32_t src_start = 0;
            uint32_t row_block_nnz_count_this_column = 0;
            for(uint32_t i = 0; i < difflist_len; ++i){
                uint32_t row_block_ind = difflist_sample_ids[i]/row_block_size;
                if(row_block_ind == x.rowblock){
                    row_block_ind--;
                }
                // Need the following things
                // 1. The starting location of the source (main_raregeno) to copy, which is src_start
                // 2. The starting location of the destination, see dest_start
                // 3. number of snps to copy row_block_nnz_count_this_column


                if ((prev_row_block_ind != row_block_ind)) {
                    // We are on the boudary of two blocks, copy snps
                    // for prev_row_block_ind
                    uint64_t dest_start = x.blk_ptr[j * x.rowblock + prev_row_block_ind] + row_block_nnz_count[prev_row_block_ind];
                    uint32_t dest_remainder = dest_start % plink2::kBitsPerWordD2;
                    uint32_t src_remainder = src_start % plink2::kBitsPerWordD2;

                    // std::cout << " dest_start is " <<  dest_start;
                    // std::cout << " src_start is " <<  src_start;
                    // std::cout << " nnz this block this column is  " <<  row_block_nnz_count_this_column << std::endl;
                    uintptr_t *rare_geno_iter = &(main_raregeno[(src_start/plink2::kBitsPerWordD2)]) - 1;
                    rare_geno_iter[0] = x.diffvec[dest_start/plink2::kBitsPerWordD2];
                    mask[0] = ((static_cast<uintptr_t>(1) << 2*dest_remainder) - 1);
                    mask[1] = ~((static_cast<uintptr_t>(1) << 2*src_remainder) - 1);
                    //uint64_t num_bit_to_transfer = 2 * (dest_remainder + plink2::kBitsPerWordD2 - src_remainder) + plink2::kBitsPerWord * plink2::DivUp((row_block_nnz_count_this_column - plink2::kBitsPerWordD2 + src_remainder)*2, plink2::kBitsPerWord);
                    uint64_t num_bit_to_transfer = 2 * (dest_remainder + row_block_nnz_count_this_column);
                    uint32_t mask_end_ind = 1;
                    uint32_t mask_end_to_keep = 2 * ( src_remainder + row_block_nnz_count_this_column);
                    if(row_block_nnz_count_this_column > (plink2::kBitsPerWordD2 - src_remainder)){
                        mask_end_ind = 2 + (row_block_nnz_count_this_column - plink2::kBitsPerWordD2 + src_remainder)/plink2::kBitsPerWordD2;
                        mask_end_to_keep = (row_block_nnz_count_this_column - plink2::kBitsPerWordD2 + src_remainder) % plink2::kBitsPerWordD2;
                        mask_end_to_keep *= 2;
                    }
                    //mask[mask_end_ind] &= ((static_cast<uintptr_t>(1) << mask_end_to_keep) - 1);
                    mask[mask_end_ind] &= (static_cast<uintptr_t>(-1) >> (plink2::kBitsPerWord - mask_end_to_keep));

                    plink2::CopyBitarrSubset(rare_geno_iter, mask, num_bit_to_transfer,&x.diffvec[dest_start/plink2::kBitsPerWordD2]);
                    mask[mask_end_ind] = static_cast<uintptr_t>(-1);



                    // after this is done update these variables
                    row_block_nnz_count[prev_row_block_ind] += row_block_nnz_count_this_column;
                    row_block_nnz_count_this_column = 0;
                    prev_row_block_ind = row_block_ind;
                    src_start = i;
                }

                
                uint64_t nnz_ind = x.blk_ptr[j * x.rowblock + row_block_ind] + row_block_nnz_count[row_block_ind] + row_block_nnz_count_this_column;
                CSB_ind p = {.row = difflist_sample_ids[i], .col = colind};
                x.index[nnz_ind] = p;
                row_block_nnz_count_this_column++;

                // What about the last iteration?
                if (i == (difflist_len - 1)) {
                    uint64_t dest_start = x.blk_ptr[j * x.rowblock + row_block_ind] + row_block_nnz_count[row_block_ind];
                    uint32_t dest_remainder = dest_start % plink2::kBitsPerWordD2;
                    uint32_t src_remainder = src_start % plink2::kBitsPerWordD2;


                    uintptr_t* rare_geno_iter = &(main_raregeno[(src_start / plink2::kBitsPerWordD2)]) - 1;
                    rare_geno_iter[0] = x.diffvec[dest_start / plink2::kBitsPerWordD2];
                    mask[0] = ((static_cast<uintptr_t>(1) << 2 * dest_remainder) - 1);
                    mask[1] = ~((static_cast<uintptr_t>(1) << 2 * src_remainder) - 1);

                    uint64_t num_bit_to_transfer = 2 * (dest_remainder + row_block_nnz_count_this_column);
                    uint32_t mask_end_ind = 1;
                    uint32_t mask_end_to_keep = 2 * (src_remainder + row_block_nnz_count_this_column);
                    if (row_block_nnz_count_this_column > (plink2::kBitsPerWordD2 - src_remainder)) {
                        mask_end_ind = 2 + (row_block_nnz_count_this_column - plink2::kBitsPerWordD2 + src_remainder) / plink2::kBitsPerWordD2;
                        mask_end_to_keep = (row_block_nnz_count_this_column - plink2::kBitsPerWordD2 + src_remainder) % plink2::kBitsPerWordD2;
                        mask_end_to_keep *= 2;
                    }

                    mask[mask_end_ind] &= (static_cast<uintptr_t>(-1) >> (plink2::kBitsPerWord - mask_end_to_keep));

                    plink2::CopyBitarrSubset(rare_geno_iter, mask, num_bit_to_transfer, &x.diffvec[dest_start / plink2::kBitsPerWordD2]);
                    mask[mask_end_ind] = static_cast<uintptr_t>(-1);

                    row_block_nnz_count[row_block_ind] += row_block_nnz_count_this_column;
                }
            }

        }
        // Finishes a stripe of vertical blocks, add some padding if needed
        for (uint32_t i = 0; i < x.rowblock; ++i) {
            uint64_t nnz_start = x.blk_ptr[j * x.rowblock + i]  + row_block_nnz_count[i];
            uint64_t nnz_end =  x.blk_ptr[j * x.rowblock + i + 1];
            std::cout << "start " << nnz_start << " end " << nnz_end << std::endl;
            for(uint64_t pad_ind = nnz_start; pad_ind < nnz_end; pad_ind++){
                x.index[pad_ind] = x.index[nnz_start - 1];
            }
            // for the diffvec
            plink2::ZeroTrailingNyps(row_block_nnz_count[i], &(x.diffvec[x.blk_ptr[j * x.rowblock + i]/plink2::kBitsPerWordD2]));
            for(uint64_t pad_ind = plink2::DivUp(nnz_start, plink2::kBitsPerWordD2); pad_ind < nnz_end/plink2::kBitsPerWordD2; ++pad_ind){
                x.diffvec[pad_ind] = static_cast<uintptr_t>(0);
            }
        }

    }



    x.loaded = true;
    plink2::aligned_free(main_raregeno_0);
    free(difflist_sample_ids);
    free(row_block_nnz_count);
    free(mask);
}


sparse_snp::sparse_snp()
{
    blk_ptr = nullptr;
    index = nullptr;
    diffvec = nullptr;
    genovec = nullptr;
    dense_ind = nullptr;
    loaded = false;
    xim = nullptr;
    rowblock = ROW_BLOCK;
    colblock = COL_BLOCK; // testing only must be greater than the number of columns
    genovec_word_ct = 0;
    ndense = 0;
}

sparse_snp::~sparse_snp() {
    if (loaded) {
        free(blk_ptr);
        free(xim);
        plink2::aligned_free(index);
        plink2::aligned_free(diffvec);
        free(dense_ind);
        plink2::aligned_free(genovec);
        std::cout << "freed " << std::endl;
    }
}

void sparse_snp::vtx(const double *v, double * result) const {
    if(!loaded){
        stop("matrix not loaded yet");
    }

    for(uint32_t i = 0; i < ni; ++i){
        result[i] = 0;
    }
    
    uint32_t local_word_ct = plink2::DivUp(no, plink2::kBitsPerWordD2);
    // Sparse columns
    #pragma omp parallel for
    for(uint32_t colblock_ind = 0; colblock_ind < colblock; ++colblock_ind){
        for(uint32_t rowblock_ind = 0; rowblock_ind < rowblock; ++rowblock_ind) {
            uint64_t start = blk_ptr[colblock_ind * rowblock + rowblock_ind];
            uint64_t end =  blk_ptr[colblock_ind * rowblock + rowblock_ind + 1];
            const uintptr_t *block_diffvec = &(diffvec[start/plink2::kBitsPerWordD2]);
            
            for(uint64_t widx = 0; widx < (end - start)/plink2::kBitsPerWordD2; ++widx){
                const uintptr_t geno_word = block_diffvec[widx];
                uintptr_t geno_word1 = geno_word & plink2::kMask5555;
                uintptr_t geno_word2 = (geno_word >> 1) & plink2::kMask5555;
                uintptr_t geno_missing_word = geno_word1 & geno_word2;
                geno_word1 ^= geno_missing_word;
                while (geno_word1) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word1) / 2;
                    // result_local += cur_weights[sample_idx_lowbits];
                    const CSB_ind local_ind = index[start + widx * plink2::kBitsPerWordD2 + sample_idx_lowbits];
                    result[local_ind.col] += v[local_ind.row];
                    geno_word1 &= geno_word1 - 1;
                }
                geno_word2 ^= geno_missing_word;
                while (geno_word2) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word2) / 2;
                    const CSB_ind local_ind = index[start + widx * plink2::kBitsPerWordD2 + sample_idx_lowbits];
                    result[local_ind.col] += 2 * v[local_ind.row];
                    geno_word2 &= geno_word2 - 1;
                }
                while (geno_missing_word) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_missing_word) / 2;
                    const CSB_ind local_ind = index[start + widx * plink2::kBitsPerWordD2 + sample_idx_lowbits];
                    result[local_ind.col] += xim[local_ind.col] * v[local_ind.row];
                    geno_missing_word &= geno_missing_word - 1;
                }
            }

        }
        uint32_t len = ndense/colblock;
        uint32_t start = len * colblock_ind;
        uint32_t end = len * (colblock_ind + 1);
        if(colblock_ind == (colblock - 1)){
            end = ndense;
        }
        for(uint32_t densecol_ind = start; densecol_ind < end; ++densecol_ind){
            uint32_t colind = dense_ind[densecol_ind];
            const uintptr_t* col = &(genovec[densecol_ind * genovec_word_ct]);
            double result_1 = 0;
            double result_2 = 0;
            double result_missing = 0;
            for (uint32_t widx = 0; widx < local_word_ct; ++widx) {
                const uintptr_t geno_word = col[widx];
                const double *cur_weights = &(v[widx * plink2::kBitsPerWordD2]);
                uintptr_t geno_word1 = geno_word & plink2::kMask5555;
                uintptr_t geno_word2 = (geno_word >> 1) & plink2::kMask5555;
                uintptr_t geno_missing_word = geno_word1 & geno_word2;
                geno_word1 ^= geno_missing_word;
                while (geno_word1) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word1) / 2;
                    result_1 +=  cur_weights[sample_idx_lowbits];
                    geno_word1 &= geno_word1 - 1;
                }
                geno_word2 ^= geno_missing_word;
                while (geno_word2) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word2) / 2;
                    result_2 +=  cur_weights[sample_idx_lowbits];
                    geno_word2 &= geno_word2 - 1;
                }
                while (geno_missing_word) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_missing_word) / 2;
                    result_missing +=  cur_weights[sample_idx_lowbits];
                    geno_missing_word &= geno_missing_word - 1;
                }
            }
            result[colind] = (result_1 + 2 * result_2 + xim[colind] * result_missing);
        }
    }
}

void sparse_snp::xv(const double *v, double * result) const {
    if(!loaded){
        stop("matrix not loaded yet");
    }
    for(uint32_t i = 0; i < no; ++i){
        result[i] = 0;
    }
    uint32_t local_word_ct = plink2::DivUp(no, plink2::kBitsPerWordD2);
    #pragma omp parallel for
    for(uint32_t rowblock_ind = 0; rowblock_ind < rowblock; ++rowblock_ind){
        for(uint32_t colblock_ind = 0; colblock_ind < colblock; ++colblock_ind) {
            uint64_t start = blk_ptr[colblock_ind * rowblock + rowblock_ind];
            uint64_t end =  blk_ptr[colblock_ind * rowblock + rowblock_ind + 1];
            const uintptr_t *block_diffvec = &(diffvec[start/plink2::kBitsPerWordD2]);
            
            for(uint64_t widx = 0; widx < (end - start)/plink2::kBitsPerWordD2; ++widx){
                const uintptr_t geno_word = block_diffvec[widx];
                uintptr_t geno_word1 = geno_word & plink2::kMask5555;
                uintptr_t geno_word2 = (geno_word >> 1) & plink2::kMask5555;
                uintptr_t geno_missing_word = geno_word1 & geno_word2;
                geno_word1 ^= geno_missing_word;
                while (geno_word1) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word1) / 2;
                    const CSB_ind local_ind = index[start + widx * plink2::kBitsPerWordD2 + sample_idx_lowbits];
                    result[local_ind.row] += v[local_ind.col];
                    geno_word1 &= geno_word1 - 1;
                }
                geno_word2 ^= geno_missing_word;
                while (geno_word2) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word2) / 2;
                    const CSB_ind local_ind = index[start + widx * plink2::kBitsPerWordD2 + sample_idx_lowbits];
                    result[local_ind.row] += 2 * v[local_ind.col];
                    geno_word2 &= geno_word2 - 1;
                }
                while (geno_missing_word) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_missing_word) / 2;
                    const CSB_ind local_ind = index[start + widx * plink2::kBitsPerWordD2 + sample_idx_lowbits];
                    result[local_ind.row] += xim[local_ind.col] * v[local_ind.col];
                    geno_missing_word &= geno_missing_word - 1;
                }
            }

        }
        // put dense genovecs here
        uint32_t len = local_word_ct/rowblock;
        uint32_t start = len * rowblock_ind;
        uint32_t end = len * (rowblock_ind + 1);
        if(rowblock_ind == (rowblock - 1)){
            end = local_word_ct;
        }
        for (uint32_t densecol_ind = 0; densecol_ind < ndense; ++densecol_ind){
            uint32_t colind = dense_ind[densecol_ind];
            const uintptr_t* col = &(genovec[densecol_ind * genovec_word_ct]);
            const double weight = v[colind];
            const double imputed_mean = xim[colind];
            for (uint32_t widx = start; widx < end; ++widx) {
                const uintptr_t geno_word = col[widx];
                uintptr_t geno_word1 = geno_word & plink2::kMask5555;
                uintptr_t geno_word2 = (geno_word >> 1) & plink2::kMask5555;
                uintptr_t geno_missing_word = geno_word1 & geno_word2;
                geno_word1 ^= geno_missing_word;
                while (geno_word1) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word1) / 2;
                    result[widx * plink2::kBitsPerWordD2 + sample_idx_lowbits] +=  weight;
                    geno_word1 &= geno_word1 - 1;
                }
                geno_word2 ^= geno_missing_word;
                while (geno_word2) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_word2) / 2;
                    result[widx * plink2::kBitsPerWordD2 + sample_idx_lowbits] +=  2 * weight;
                    geno_word2 &= geno_word2 - 1;
                }
                while (geno_missing_word) {
                    const uint32_t sample_idx_lowbits = plink2::ctzw(geno_missing_word) / 2;
                    result[widx * plink2::kBitsPerWordD2 + sample_idx_lowbits] +=  imputed_mean * weight;
                    geno_missing_word &= geno_missing_word - 1;
                }
            }
        }
    }


}

uint32_t sparse_snp::Getnrow() const {
    return no;
}

uint32_t sparse_snp::Getncol() const {
    return ni;
}

// [[Rcpp::export]]
SEXP NewSparse(List pgen, IntegerVector variant_subset) {
   if (strcmp_r_c(pgen[0], "pgen")) {
    stop("pgen is not a pgen object");
  }
  XPtr<class RPgenReader> rp = as<XPtr<class RPgenReader> >(pgen[1]);
  XPtr<class sparse_snp> xp(new sparse_snp(), true);
  rp->LoadSparse(*xp, &variant_subset[0], variant_subset.size());
  return List::create(_["class"] = "sparse_snp", _["sparse_snp"] = xp);
}

// [[Rcpp::export]]
NumericVector SparseTest(List mat, NumericVector v) {
  if (strcmp_r_c(mat[0], "sparse_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class sparse_snp> x = as<XPtr<class sparse_snp> >(mat[1]);
  if(v.size() != x->Getnrow()){
      stop("vector length incompatible with the matrix.");
  }
  NumericVector result(x->Getncol());
  x->vtx(&v[0], &result[0]);
  return result;
}

// [[Rcpp::export]]
NumericVector SparseTest2(List mat, NumericVector v) {
  if (strcmp_r_c(mat[0], "sparse_snp")) {
    stop("matrix not the right type");
  }
  XPtr<class sparse_snp> x = as<XPtr<class sparse_snp> >(mat[1]);
  if(v.size() != x->Getncol()){
      stop("vector length incompatible with the matrix.");
  }
  NumericVector result(x->Getnrow());
  x->xv(&v[0], &result[0]);
  return result;
}