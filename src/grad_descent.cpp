#include <tsne/grad_descent.h>
#include <tsne/baseline.h>
#include <immintrin.h>

static void printvec(__m256d v){
    double p[4] = {0,0,0,0};
    _mm256_store_pd(p, v);
    printf("vec: %.10f %.10f %.10f %.10f\n", p[0],p[1], p[2], p[3]);
}

static int check(__m256d v, double d1, double d2, double d3, double d4){
    double p[4] = {0,0,0,0};
    _mm256_store_pd(p, v);
    if(p[0] != d1){ printf("First: %.10f != %.10f!\n", p[0], d1); return 1;}
    if(p[1] != d2){ printf("Second: %.10f != %.10f!\n", p[1], d2); return 1;}
    if(p[2] != d3){ printf("Third: %.10f != %.10f!\n", p[2], d3); return 1;}
    if(p[3] != d4){ printf("Fourth: %.10f != %.10f!\n", p[3], d4); return 1;}
    return 0;
}

void grad_desc_b(Matrix *Y, tsne_var_t *var, int n, int n_dim,
                 double momentum)
{
    // calculate low-dimensional affinities
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

    // calculate gradient with respect to embeddings Y
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            double value = (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
                           var->Q_numerators.data[i * n + j];
            var->tmp.data[i * n + j] = value;
            var->tmp.data[j * n + i] = value;
        }
        var->tmp.data[i * n + i] = 0.0;
    }
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n_dim; k++)
        {
            double value = 0;
            for (int j = 0; j < n; j++)
            {
                value += var->tmp.data[i * n + j] *
                         (Y->data[i * n_dim + k] - Y->data[j * n_dim + k]);
            }
            value *= 4;
            var->grad_Y.data[i * n_dim + k] = value;
        }
    }

    // calculate gains, according to adaptive heuristic of Python implementation
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            bool positive_grad = (var->grad_Y.data[i * n_dim + j] > 0);
            bool positive_delta = (var->Y_delta.data[i * n_dim + j] > 0);
            double value = var->gains.data[i * n_dim + j];
            if ((positive_grad && positive_delta) ||
                (!positive_grad && !positive_delta))
            {
                value *= 0.8;
            }
            else
            {
                value += 0.2;
            }
            if (value < kMinGain)
                value = kMinGain;
            var->gains.data[i * n_dim + j] = value;
        }
    }

    // update step
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            double value =
                momentum * var->Y_delta.data[i * n_dim + j] -
                kEta * var->gains.data[i * n_dim + j] * var->grad_Y.data[i * n_dim + j];
            var->Y_delta.data[i * n_dim + j] = value;
            Y->data[i * n_dim + j] += value;
        }
    }

    // center each dimension at 0
    double means[n_dim];
    for (int j = 0; j < n_dim; j++)
    {
        means[j] = 0;
    }
    // accumulate
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            means[j] += Y->data[i * n_dim + j];
        }
    }
    // take mean
    for (int j = 0; j < n_dim; j++)
    {
        means[j] /= n;
    }
    // center
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n_dim; j++)
        {
            Y->data[i * n_dim + j] -= means[j];
        }
    }
}

void grad_desc_ndim_unroll(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

    // calculate gradient with respect to embeddings Y
    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            double value = (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
                           var->Q_numerators.data[i * n + j];
            var->tmp.data[i * n + j] = value;
            var->tmp.data[j * n + i] = value;
        }
        var->tmp.data[i * n + i] = 0.0;
    }
    int twoi = 0;
    for (int i = 0; i < n; i++){
        double value0 = 0;
        double value1 = 0;
        int twoj = 0;
        for (int j = 0; j < n; j++){
            value0 += var->tmp.data[i * n + j] * (Y->data[twoi] - Y->data[twoj]);
            value1 += var->tmp.data[i * n + j] * (Y->data[twoi + 1] - Y->data[twoj + 1]);
            twoj += 2;
        }
        value0 *= 4;
        value1 *= 4;
        var->grad_Y.data[twoi] = value0;
        var->grad_Y.data[twoi + 1] = value1;
        twoi += 2;
    }

    // calculate gains, according to adaptive heuristic of Python implementation
    for (int i = 0; i < n*2; i+=2){
        bool positive_grad0 = (var->grad_Y.data[i] > 0);
        bool positive_delta0 = (var->Y_delta.data[i] > 0);
        bool positive_grad1 = (var->grad_Y.data[i+1] > 0);
        bool positive_delta1 = (var->Y_delta.data[i+1] > 0);
        double value0 = var->gains.data[i];
        double value1 = var->gains.data[i+1];

        if ((positive_grad0 && positive_delta0) || (!positive_grad0 && !positive_delta0)){
            value0 *= 0.8;
        }
        else{
            value0 += 0.2;
        }
        if (value0 < kMinGain){
            value0 = kMinGain;
        }
        var->gains.data[i] = value0;

        if ((positive_grad1 && positive_delta1) || (!positive_grad1 && !positive_delta1)){
            value1 *= 0.8;
        }
        else{
            value1 += 0.2;
        }
        if (value1 < kMinGain){
            value1 = kMinGain;
        }
        var->gains.data[i+1] = value1;
    }

    // update step
    for (int i = 0; i < n*2; i+= 2){
        double value0 = momentum * var->Y_delta.data[i] -
            kEta * var->gains.data[i] * var->grad_Y.data[i];
        var->Y_delta.data[i] = value0;
        Y->data[i] += value0;

        double value1 = momentum * var->Y_delta.data[i + 1] -
            kEta * var->gains.data[i + 1] * var->grad_Y.data[i + 1];
        var->Y_delta.data[i + 1] = value1;
        Y->data[i + 1] += value1;
    }

    // center each dimension at 0
    double means[2] = {0};

    // accumulate
    for (int i = 0; i < n*2; i+=2){
        means[0] += Y->data[i];
        means[1] += Y->data[i + 1];
    }
    // take mean
    means[0] /= n;
    means[1] /= n;
    // center
    for (int i = 0; i < n*2; i+=2){
        Y->data[i] -= means[0];
        Y->data[i + 1] -= means[1];
    }
}

void grad_desc_mean_unroll(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

    // calculate gradient with respect to embeddings Y
    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            double value = (var->P.data[i * n + j] - var->Q.data[i * n + j]) *
                           var->Q_numerators.data[i * n + j];
            var->tmp.data[i * n + j] = value;
            var->tmp.data[j * n + i] = value;
        }
        var->tmp.data[i * n + i] = 0.0;
    }
    int twoi = 0;
    for (int i = 0; i < n; i++){
        double value0 = 0;
        double value1 = 0;
        int twoj = 0;
        for (int j = 0; j < n; j++){
            value0 += var->tmp.data[i * n + j] * (Y->data[twoi] - Y->data[twoj]);
            value1 += var->tmp.data[i * n + j] * (Y->data[twoi + 1] - Y->data[twoj + 1]);
            twoj += 2;
        }
        value0 *= 4;
        value1 *= 4;
        var->grad_Y.data[twoi] = value0;
        var->grad_Y.data[twoi + 1] = value1;
        twoi += 2;
    }

    // calculate gains, according to adaptive heuristic of Python implementation
    for (int i = 0; i < n*2; i+=2){
        bool positive_grad0 = (var->grad_Y.data[i] > 0);
        bool positive_delta0 = (var->Y_delta.data[i] > 0);
        bool positive_grad1 = (var->grad_Y.data[i+1] > 0);
        bool positive_delta1 = (var->Y_delta.data[i+1] > 0);
        double value0 = var->gains.data[i];
        double value1 = var->gains.data[i+1];

        if ((positive_grad0 && positive_delta0) || (!positive_grad0 && !positive_delta0)){
            value0 *= 0.8;
        }
        else{
            value0 += 0.2;
        }
        if (value0 < kMinGain){
            value0 = kMinGain;
        }
        var->gains.data[i] = value0;

        if ((positive_grad1 && positive_delta1) || (!positive_grad1 && !positive_delta1)){
            value1 *= 0.8;
        }
        else{
            value1 += 0.2;
        }
        if (value1 < kMinGain){
            value1 = kMinGain;
        }
        var->gains.data[i+1] = value1;
    }

    // update step
    for (int i = 0; i < n*2; i+= 2){
        double value0 = momentum * var->Y_delta.data[i] -
            kEta * var->gains.data[i] * var->grad_Y.data[i];
        var->Y_delta.data[i] = value0;
        Y->data[i] += value0;

        double value1 = momentum * var->Y_delta.data[i + 1] -
            kEta * var->gains.data[i + 1] * var->grad_Y.data[i + 1];
        var->Y_delta.data[i + 1] = value1;
        Y->data[i + 1] += value1;
    }

    // center each dimension at 0
    double mean0 = 0, mean1 = 0;

    // accumulate
    for (int i = 0; i < n*2; i+=2){
        mean0 += Y->data[i];
        mean1 += Y->data[i + 1];
    }
    // take mean
    mean0 /= n;
    mean1 /= n;
    // center
    for (int i = 0; i < n*2; i+=2){
        Y->data[i] -= mean0;
        Y->data[i + 1] -= mean1;
    }
}

void grad_desc_tmp_opt(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    //calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
    double *pdata = var->P.data;
    double *qdata = var->Q.data;
    double *q_numdata = var->Q_numerators.data;
    double *ydata = Y->data;
    double *gainsdata = var->gains.data;
    double *gradydata = var->grad_Y.data;
    double *ydeltadata = var->Y_delta.data;

    // calculate gradient with respect to embeddings Y
    int twoi = 0;
    for (int i = 0; i < n; i++){
        double value0 = 0;
        double value1 = 0;
        int twoj = 0;
        double ydatatwoi = ydata[twoi];
        double ydatatwoip1 = ydata[twoi+1];
        double tmp;
        for (int j = 0; j < n; j++){
            tmp = (pdata[i * n + j] - qdata[i * n + j]) * q_numdata[i * n + j];
            value0 += tmp * (ydatatwoi - ydata[twoj]);
            value1 += tmp * (ydatatwoip1 - ydata[twoj + 1]);
            twoj += 2;
        }
        value0 *= 4;
        value1 *= 4;
        gradydata[twoi] = value0;
        gradydata[twoi + 1] = value1;
        twoi += 2;
    }

    // calculate gains, according to adaptive heuristic of Python implementation
    for (int i = 0; i < n*2; i+=2){
        bool positive_grad0 = (gradydata[i] > 0);
        bool positive_delta0 = (ydeltadata[i] > 0);
        bool positive_grad1 = (gradydata[i+1] > 0);
        bool positive_delta1 = (ydeltadata[i+1] > 0);
        double value0 = gainsdata[i];
        double value1 = gainsdata[i+1];

        value0 = (positive_grad0 == positive_delta0) ? value0 * 0.8 : value0 + 0.2;
        if (value0 < kMinGain){
            value0 = kMinGain;
        }
        gainsdata[i] = value0;

        value1 = (positive_grad1 == positive_delta1) ? value1 * 0.8 : value1 + 0.2;
        if (value1 < kMinGain){
            value1 = kMinGain;
        }
        gainsdata[i+1] = value1;
    }

    // update step
    double mean0 = 0, mean1 = 0;
    for (int i = 0; i < n*2; i+= 2){
        double value0 = momentum * ydeltadata[i] -
            kEta * gainsdata[i] * gradydata[i];
        ydeltadata[i] = value0;
        ydata[i] += value0;
        mean0 += ydata[i];

        double value1 = momentum * ydeltadata[i + 1] -
            kEta * gainsdata[i + 1] * gradydata[i + 1];
        ydeltadata[i + 1] = value1;
        ydata[i+1] += value1;
        mean1 += ydata[i+1];
    }

    // center each dimension at 0

    // take mean
    mean0 /= n;
    mean1 /= n;
    // center
    for (int i = 0; i < n*2; i+=2){
        ydata[i] -= mean0;
        ydata[i + 1] -= mean1;
    }
}

void grad_desc_loop_merge(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    //calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
    double *pdata = var->P.data;
    double *qdata = var->Q.data;
    double *q_numdata = var->Q_numerators.data;
    double *ydata = Y->data;
    double *gainsdata = var->gains.data;
    double *gradydata = var->grad_Y.data;
    double *ydeltadata = var->Y_delta.data;

    // calculate gradient with respect to embeddings Y
    int twoi = 0;
    for (int i = 0; i < n; i++){
        double value0 = 0;
        double value1 = 0;
        int twoj = 0;
        double ydatatwoi = ydata[twoi];
        double ydatatwoip1 = ydata[twoi+1];
        double tmp;
        for (int j = 0; j < n; j++){
            tmp = (pdata[i * n + j] - qdata[i * n + j]) * q_numdata[i * n + j];
            value0 += tmp * (ydatatwoi - ydata[twoj]);
            value1 += tmp * (ydatatwoip1 - ydata[twoj + 1]);
            twoj += 2;
        }
        value0 *= 4;
        value1 *= 4;
        gradydata[twoi] = value0;
        gradydata[twoi + 1] = value1;

        // calculate gains, according to adaptive heuristic of Python implementation
        bool positive_grad0 = (value0 > 0);
        bool positive_delta0 = (ydeltadata[twoi] > 0);
        bool positive_grad1 = (value1 > 0);
        bool positive_delta1 = (ydeltadata[twoi+1] > 0);
        double val0 = gainsdata[twoi];
        double val1 = gainsdata[twoi+1];

        val0 = (positive_grad0 == positive_delta0) ? val0 * 0.8 : val0 + 0.2;
        val1 = (positive_grad1 == positive_delta1) ? val1 * 0.8 : val1 + 0.2;
        if (val0 < kMinGain){
            val0 = kMinGain;
        }
        if (val1 < kMinGain){
            val1 = kMinGain;
        }

        gainsdata[twoi] = val0;
        gainsdata[twoi+1] = val1;

        twoi += 2;
    }

    twoi = 0;
    double mean0 = 0, mean1 = 0;
    for(int i=0; i<n; i++){
        //update step
        double v0 = momentum * ydeltadata[twoi] - kEta * gainsdata[twoi] * gradydata[twoi];
        double v1 = momentum * ydeltadata[twoi+1] - kEta * gainsdata[twoi+1] * gradydata[twoi+1];
        ydeltadata[twoi] = v0;
        ydeltadata[twoi + 1] = v1;
        ydata[twoi] += v0;
        ydata[twoi+1] += v1;
        mean0 += ydata[twoi];
        mean1 += ydata[twoi+1];
        twoi+=2;
    }
    // take mean
    mean0 /= n;
    mean1 /= n;
    // center
    for (int i = 0; i < n*2; i+=2){
        ydata[i] -= mean0;
        ydata[i + 1] -= mean1;
    }
}

//from here assumes n is even

void grad_desc_accumulators2(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    //calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
    double *pdata = var->P.data;
    double *qdata = var->Q.data;
    double *q_numdata = var->Q_numerators.data;
    double *ydata = Y->data;
    double *gainsdata = var->gains.data;
    double *gradydata = var->grad_Y.data;
    double *ydeltadata = var->Y_delta.data;

    //calculate gradient with respect to embeddings Y
    int twoi = 0;
    for (int i = 0; i < n; i+=2){
        double value0 = 0, value1 = 0, value2 = 0, value3 = 0;
        int twoj = 0;
        double ydatatwoi = ydata[twoi], ydatatwoip1 = ydata[twoi+1];
        double tmp1, tmp2;
        for (int j = 0; j < n; j+=2){
            tmp1 = (pdata[i*n+j] - qdata[i*n+j]) * q_numdata[i*n+j];
            tmp2 = (pdata[i*n+j+1] - qdata[i*n+j+1]) * q_numdata[i*n+j+1];
            value0 += tmp1 * (ydatatwoi - ydata[twoj]);
            value1 += tmp1 * (ydatatwoip1 - ydata[twoj+1]);
            value2 += tmp2 * (ydatatwoi - ydata[twoj+2]);
            value3 += tmp2 * (ydatatwoip1 - ydata[twoj+3]);
            twoj += 4;
        }
        value0 += value2;
        value1 += value3;
        value0 *= 4;
        value1 *= 4;
        gradydata[twoi] = value0;
        gradydata[twoi+1] = value1;


        value2 = 0, value3 = 0;
        double value4=0,value5=0;
        twoj = 0;
        ydatatwoi = ydata[twoi+2], ydatatwoip1 = ydata[twoi+3];
        for (int j = n; j < 2*n; j+=2){
            tmp1 = (pdata[i*n+j] - qdata[i*n+j]) * q_numdata[i*n+j];
            tmp2 = (pdata[i*n+j+1] - qdata[i*n+j+1]) * q_numdata[i*n+j+1];
            value2 += tmp1 * (ydatatwoi - ydata[twoj]);
            value3 += tmp1 * (ydatatwoip1 - ydata[twoj+1]);
            value4 += tmp2 * (ydatatwoi - ydata[twoj+2]);
            value5 += tmp2 * (ydatatwoip1 - ydata[twoj+3]);
            twoj += 4;
        }
        value2 += value4;
        value3 += value5;
        value2 *= 4;
        value3 *= 4;
        gradydata[twoi+2] = value2;
        gradydata[twoi+3] = value3;


        // calculate gains, according to adaptive heuristic of Python implementation
        bool positive_grad0 = ((value0) > 0);
        bool positive_delta0 = (ydeltadata[twoi] > 0);
        bool positive_grad1 = ((value1) > 0);
        bool positive_delta1 = (ydeltadata[twoi+1] > 0);
        bool positive_grad2 = ((value2) > 0);
        bool positive_delta2 = (ydeltadata[twoi+2] > 0);
        bool positive_grad3 = ((value3) > 0);
        bool positive_delta3 = (ydeltadata[twoi+3] > 0);
        double val0 = gainsdata[twoi];
        double val1 = gainsdata[twoi+1];
        double val2 = gainsdata[twoi+2];
        double val3 = gainsdata[twoi+3];

        val0 = (positive_grad0 == positive_delta0) ? val0 * 0.8 : val0 + 0.2;
        val1 = (positive_grad1 == positive_delta1) ? val1 * 0.8 : val1 + 0.2;
        val2 = (positive_grad2 == positive_delta2) ? val2 * 0.8 : val2 + 0.2;
        val3 = (positive_grad3 == positive_delta3) ? val3 * 0.8 : val3 + 0.2;
        if (val0 < kMinGain){ val0 = kMinGain; }
        if (val1 < kMinGain){ val1 = kMinGain; }
        if (val2 < kMinGain){ val2 = kMinGain; }
        if (val3 < kMinGain){ val3 = kMinGain; }

        gainsdata[twoi] = val0;
        gainsdata[twoi+1] = val1;
        gainsdata[twoi+2] = val2;
        gainsdata[twoi+3] = val3;

        twoi += 4;
    }

    double mean0 = 0, mean1 = 0, mean2=0, mean3=0;
    int twon = 2*n;
    for(int i=0; i<twon; i+=4){
        //update step
        double v0 = momentum * ydeltadata[i] - kEta * gainsdata[i] * gradydata[i];
        double v1 = momentum * ydeltadata[i+1] - kEta * gainsdata[i+1] * gradydata[i+1];
        double v2 = momentum * ydeltadata[i+2] - kEta * gainsdata[i+2] * gradydata[i+2];
        double v3 = momentum * ydeltadata[i+3] - kEta * gainsdata[i+3] * gradydata[i+3];
        ydeltadata[i] = v0;
        ydeltadata[i+1] = v1;
        ydeltadata[i+2] = v2;
        ydeltadata[i+3] = v3;
        ydata[i] += v0;
        ydata[i+1] += v1;
        ydata[i+2] += v2;
        ydata[i+3] += v3;
        mean0 += ydata[i];
        mean1 += ydata[i+1];
        mean2 += ydata[i+2];
        mean3 += ydata[i+3];
    }
    // take mean
    mean0 += mean2;
    mean1 += mean3;
    mean0 /= n;
    mean1 /= n;
    // center
    for (int i = 0; i < twon; i+=4){
        ydata[i] -= mean0;
        ydata[i+1] -= mean1;
        ydata[i+2] -= mean0;
        ydata[i+3] -= mean1;
    }
}

void grad_desc_accumulators(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
    double *pdata = var->P.data;
    double *qdata = var->Q.data;
    double *q_numdata = var->Q_numerators.data;
    double *ydata = Y->data;
    double *gainsdata = var->gains.data;
    double *ydeltadata = var->Y_delta.data;

    //calculate gradient with respect to embeddings Y
    int twoi = 0;
    for (int i = 0; i < n; i++){
        double value0 = 0;
        double value1 = 0;
        double value2 = 0;
        double value3 = 0;
        int twoj = 0;
        double ydatatwoi = ydata[twoi];
        double ydatatwoip1 = ydata[twoi+1];
        double tmp1, tmp2;
        //try blocking approach?
        for (int j = 0; j < n; j+=2){
            tmp1 = (pdata[i * n + j] - qdata[i * n + j]) * q_numdata[i * n + j];
            tmp2 = (pdata[i * n + j+1] - qdata[i * n + j+1]) * q_numdata[i * n + j+1];
            value0 += tmp1 * (ydatatwoi - ydata[twoj]);
            value1 += tmp1 * (ydatatwoip1 - ydata[twoj + 1]);
            value2 += tmp2 * (ydatatwoi - ydata[twoj+2]);
            value3 += tmp2 * (ydatatwoip1 - ydata[twoj + 3]);
            twoj += 4;
        }
        value0 += value2;
        value1 += value3;

        // calculate gains, according to adaptive heuristic of Python implementation
        double ydeltadata2i = ydeltadata[twoi];
        double ydeltadata2ip1 = ydeltadata[twoi+1];
        bool positive_grad0 = ((value0) > 0);
        bool positive_delta0 = (ydeltadata2i > 0);
        bool positive_grad1 = ((value1) > 0);
        bool positive_delta1 = (ydeltadata2ip1 > 0);
        double val0 = gainsdata[twoi];
        double val1 = gainsdata[twoi+1];

        val0 = (positive_grad0 == positive_delta0) ? val0 * 0.8 : val0 + 0.2;
        val1 = (positive_grad1 == positive_delta1) ? val1 * 0.8 : val1 + 0.2;
        if (val0 < kMinGain){ val0 = kMinGain; }
        if (val1 < kMinGain){ val1 = kMinGain; }

        gainsdata[twoi] = val0;
        gainsdata[twoi+1] = val1;
        ydeltadata[twoi] = momentum * ydeltadata2i - fourkEta * val0 * value0;
        ydeltadata[twoi+1] = momentum * ydeltadata2ip1 - fourkEta * val1 * value1;

        twoi += 2;
    }

    double mean0 = 0, mean1 = 0, mean2=0, mean3=0;
    int twon = 2*n;
    for(int i=0; i<twon; i+=4){
        //update step
        ydata[i] += ydeltadata[i];
        ydata[i+1] += ydeltadata[i+1];
        ydata[i+2] += ydeltadata[i+2];
        ydata[i+3] += ydeltadata[i+3];
        mean0 += ydata[i];
        mean1 += ydata[i+1];
        mean2 += ydata[i+2];
        mean3 += ydata[i+3];
    }
    // take mean
    mean0 += mean2;
    mean1 += mean3;
    mean0 /= n;
    mean1 /= n;
    // center
    for (int i = 0; i < twon; i+=4){
        ydata[i] -= mean0;
        ydata[i+1] -= mean1;
        ydata[i+2] -= mean0;
        ydata[i+3] -= mean1;
    }
}

void grad_desc_vec_bottom(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
    double *pdata = var->P.data;
    double *qdata = var->Q.data;
    double *q_numdata = var->Q_numerators.data;
    double *ydata = Y->data;
    double *gainsdata = var->gains.data;
    double *ydeltadata = var->Y_delta.data;

    //calculate gradient with respect to embeddings Y
    int twoi = 0;
    for (int i = 0; i < n; i++){
        double value0 = 0;
        double value1 = 0;
        double value2 = 0;
        double value3 = 0;
        int twoj = 0;
        double ydatatwoi = ydata[twoi];
        double ydatatwoip1 = ydata[twoi+1];
        double tmp1, tmp2;
        //try blocking approach?
        for (int j = 0; j < n; j+=2){
            tmp1 = (pdata[i * n + j] - qdata[i * n + j]) * q_numdata[i * n + j];
            tmp2 = (pdata[i * n + j+1] - qdata[i * n + j+1]) * q_numdata[i * n + j+1];
            value0 += tmp1 * (ydatatwoi - ydata[twoj]);
            value1 += tmp1 * (ydatatwoip1 - ydata[twoj + 1]);
            value2 += tmp2 * (ydatatwoi - ydata[twoj+2]);
            value3 += tmp2 * (ydatatwoip1 - ydata[twoj + 3]);
            twoj += 4;
        }
        value0 += value2;
        value1 += value3;

        // calculate gains, according to adaptive heuristic of Python implementation
        double ydeltadata2i = ydeltadata[twoi];
        double ydeltadata2ip1 = ydeltadata[twoi+1];
        bool positive_grad0 = ((value0) > 0);
        bool positive_delta0 = (ydeltadata2i > 0);
        bool positive_grad1 = ((value1) > 0);
        bool positive_delta1 = (ydeltadata2ip1 > 0);
        double val0 = gainsdata[twoi];
        double val1 = gainsdata[twoi+1];

        val0 = (positive_grad0 == positive_delta0) ? val0 * 0.8 : val0 + 0.2;
        val1 = (positive_grad1 == positive_delta1) ? val1 * 0.8 : val1 + 0.2;
        if (val0 < kMinGain){ val0 = kMinGain; }
        if (val1 < kMinGain){ val1 = kMinGain; }

        gainsdata[twoi] = val0;
        gainsdata[twoi+1] = val1;
        ydeltadata[twoi] = momentum * ydeltadata2i - fourkEta * val0 * value0;
        ydeltadata[twoi+1] = momentum * ydeltadata2ip1 - fourkEta * val1 * value1;

        twoi += 2;
    }

    __m256d mean = {0,0,0,0};
    __m256d n_vec = {n,n,n,n};
    __m256d y, ydelta, mean_shuffled, mean_shuffled2;
    int mean_mask = 0b10001101; //setup such that we have [mean1 mean3 mean0 mean2]
    int mean_mask2 = 0b11011000; //setup such that we have [mean0 mean2 mean1 mean3]
    int mean_last_mask = 0b0110;
    int twon = 2*n;
    for(int i=0; i<twon; i+=4){
        //load y, ydelta
        y = _mm256_load_pd(ydata+i);
        ydelta = _mm256_load_pd(ydeltadata+i);
        
        //update step
        y = _mm256_add_pd(y, ydelta);
        mean = _mm256_add_pd(mean, y);

        _mm256_store_pd(ydata+i, y);
    }
    mean_shuffled = _mm256_permute4x64_pd(mean, mean_mask);
    mean_shuffled2 = _mm256_permute4x64_pd(mean, mean_mask2);
    mean = _mm256_hadd_pd(mean_shuffled2, mean_shuffled); //mean now holds [mean0 mean1 mean1 mean0]
    mean = _mm256_permute_pd(mean, mean_last_mask);
    // take mean
    mean = _mm256_div_pd(mean, n_vec);
    // center
    for (int i = 0; i < twon; i+=4){
        y = _mm256_load_pd(ydata+i);
        y = _mm256_sub_pd(y, mean);
        _mm256_store_pd(ydata+i, y);
    }
}

void grad_desc_vectorized(Matrix *Y, tsne_var_t *var, int n, int n_dim, double momentum){
    // calculate low-dimensional affinities
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);

    int ymask = 0b11011000; //switch elements at pos 1 and 2
    double *pdata = var->P.data;
    double *qdata = var->Q.data;
    double *q_numdata = var->Q_numerators.data;
    double *ydata = Y->data;
    double *gainsdata = var->gains.data;
    double *ydeltadata = var->Y_delta.data;

    //calculate gradient with respect to embeddings Y
    int twoi = 0;
    __m256d zero = {0,0,0,0};
    for (int i = 0; i < n; i+=4){
        int twoj = 0;
        __m256d yleft, yright, y1, y2, yfixleft1, yfixleft2, yfixleft3, yfixleft4, yfixright1, yfixright2, yfixright3, yfixright4;
        __m256d valueleft1 = _mm256_setzero_pd();
        __m256d valueright1 = _mm256_setzero_pd();
        __m256d valueleft2 = _mm256_setzero_pd();
        __m256d valueright2 = _mm256_setzero_pd();
        __m256d valueleft3 = _mm256_setzero_pd();
        __m256d valueright3 = _mm256_setzero_pd();
        __m256d valueleft4 = _mm256_setzero_pd();
        __m256d valueright4 = _mm256_setzero_pd();
        yfixleft1 = _mm256_broadcast_sd(ydata+twoi);
        yfixleft2 = _mm256_broadcast_sd(ydata+twoi+2);
        yfixleft3 = _mm256_broadcast_sd(ydata+twoi+4);
        yfixleft4 = _mm256_broadcast_sd(ydata+twoi+6);
        yfixright1 = _mm256_broadcast_sd(ydata+twoi+1);
        yfixright2 = _mm256_broadcast_sd(ydata+twoi+3);
        yfixright3 = _mm256_broadcast_sd(ydata+twoi+5);
        yfixright4 = _mm256_broadcast_sd(ydata+twoi+7);
        for (int j = 0; j < n; j+=4){
            __m256d p1, p2, p3, p4, q1, q2, q3, q4, qnum1, qnum2, qnum3, qnum4, tmp1, tmp2, tmp3, tmp4;
            y1 = _mm256_load_pd(ydata+twoj);
            y2 = _mm256_load_pd(ydata+twoj+4);
            //sort such that we have column wise 4 y elements
            yleft = _mm256_unpacklo_pd(y1, y2);
            yright = _mm256_unpackhi_pd(y1, y2);
            yleft = _mm256_permute4x64_pd(yleft, ymask);
            yright = _mm256_permute4x64_pd(yright, ymask);
            p1 = _mm256_load_pd(pdata+i*n+j);
            p2 = _mm256_load_pd(pdata+i*n+j+n);
            p3 = _mm256_load_pd(pdata+i*n+j+2*n);
            p4 = _mm256_load_pd(pdata+i*n+j+3*n);
            q1 = _mm256_load_pd(qdata+i*n+j);
            q2 = _mm256_load_pd(qdata+i*n+j+n);
            q3 = _mm256_load_pd(qdata+i*n+j+2*n);
            q4 = _mm256_load_pd(qdata+i*n+j+3*n);
            qnum1 = _mm256_load_pd(q_numdata+i*n+j);
            qnum2 = _mm256_load_pd(q_numdata+i*n+j+n);
            qnum3 = _mm256_load_pd(q_numdata+i*n+j+2*n);
            qnum4 = _mm256_load_pd(q_numdata+i*n+j+3*n);
            
            tmp1 = _mm256_mul_pd(_mm256_sub_pd(p1, q1), qnum1);
            tmp2 = _mm256_mul_pd(_mm256_sub_pd(p2, q2), qnum2);
            tmp3 = _mm256_mul_pd(_mm256_sub_pd(p3, q3), qnum3);
            tmp4 = _mm256_mul_pd(_mm256_sub_pd(p4, q4), qnum4);
            valueleft1 = _mm256_add_pd(_mm256_mul_pd(tmp1, _mm256_sub_pd(yfixleft1, yleft)), valueleft1);
            valueleft2 = _mm256_add_pd(_mm256_mul_pd(tmp2, _mm256_sub_pd(yfixleft2, yleft)), valueleft2);
            valueleft3 = _mm256_add_pd(_mm256_mul_pd(tmp3, _mm256_sub_pd(yfixleft3, yleft)), valueleft3);
            valueleft4 = _mm256_add_pd(_mm256_mul_pd(tmp4, _mm256_sub_pd(yfixleft4, yleft)), valueleft4);
            valueright1 = _mm256_add_pd(_mm256_mul_pd(tmp1, _mm256_sub_pd(yfixright1, yright)), valueright1);
            valueright2 = _mm256_add_pd(_mm256_mul_pd(tmp2, _mm256_sub_pd(yfixright2, yright)), valueright2);
            valueright3 = _mm256_add_pd(_mm256_mul_pd(tmp3, _mm256_sub_pd(yfixright3, yright)), valueright3);
            valueright4 = _mm256_add_pd(_mm256_mul_pd(tmp4, _mm256_sub_pd(yfixright4, yright)), valueright4);
            twoj += 8;
        }
        
        double *v = (double *)aligned_alloc(32, 16*sizeof(double));
        _mm256_store_pd(v, _mm256_hadd_pd(valueleft1, valueright1));
        _mm256_store_pd(v+4, _mm256_hadd_pd(valueleft2, valueright2));
        _mm256_store_pd(v+8, _mm256_hadd_pd(valueleft3, valueright3));
        _mm256_store_pd(v+12, _mm256_hadd_pd(valueleft4, valueright4));
        __m256d values_left, values_right;
        values_left = _mm256_set_pd(v[12]+v[14], v[8]+v[10], v[4]+v[6], v[0]+v[2]);
        values_right = _mm256_set_pd(v[13]+v[15], v[9]+v[11], v[5]+v[7], v[1]+v[3]);


        __m256d ydeltaleft, ydeltaright, ydelta1, ydelta2, pos_grad_left, 
        pos_grad_right, pos_delta_left, pos_delta_right, gainsleft, gainsright, gains0, gains1;
        ydelta1 = _mm256_load_pd(ydeltadata+twoi);
        ydelta2 = _mm256_load_pd(ydeltadata+twoi+4);
        //sort such that we have column wise 4 ydelta elements
        ydeltaleft = _mm256_unpacklo_pd(ydelta1, ydelta2);
        ydeltaright = _mm256_unpackhi_pd(ydelta1, ydelta2);
        ydeltaleft = _mm256_permute4x64_pd(ydeltaleft, ymask);
        ydeltaright = _mm256_permute4x64_pd(ydeltaright, ymask);

        //compute boolean masks
        pos_grad_left = _mm256_cmp_pd(values_left, zero, _CMP_GT_OQ);
        pos_grad_right = _mm256_cmp_pd(values_right, zero, _CMP_GT_OQ);
        pos_delta_left = _mm256_cmp_pd(ydeltaleft, zero, _CMP_GT_OQ);
        pos_delta_right = _mm256_cmp_pd(ydeltaright, zero, _CMP_GT_OQ);

        //load gains
        gains0 = _mm256_load_pd(gainsdata+twoi);
        gains1 = _mm256_load_pd(gainsdata+twoi+4);

        //sort gains into left and right
        gainsleft = _mm256_unpacklo_pd(gains0, gains1);
        gainsright = _mm256_unpackhi_pd(gains0, gains1);
        gainsleft = _mm256_permute4x64_pd(gainsleft, ymask);
        gainsright = _mm256_permute4x64_pd(gainsright, ymask);

        __m256d gainsmul_left, gainsmul_right, gainsplus_left, gainsplus_right, mask_left, mask_right;
        __m256d mulconst = {0.8, 0.8, 0.8, 0.8};
        __m256d addconst = {0.2, 0.2, 0.2, 0.2};
        gainsmul_left = _mm256_mul_pd(gainsleft, mulconst);
        gainsmul_right = _mm256_mul_pd(gainsright, mulconst);
        gainsplus_left = _mm256_add_pd(gainsleft, addconst);
        gainsplus_right = _mm256_add_pd(gainsright, addconst);
        mask_left = _mm256_cmp_pd(pos_grad_left, pos_delta_left, _CMP_EQ_OQ);
        mask_right = _mm256_cmp_pd(pos_grad_right, pos_delta_right, _CMP_EQ_OQ);

        gainsmul_left = _mm256_and_pd(mask_left, gainsmul_left);
        gainsmul_right = _mm256_and_pd(mask_right, gainsmul_right);
        gainsplus_left = _mm256_andnot_pd(mask_left, gainsplus_left);
        gainsplus_right = _mm256_andnot_pd(mask_right, gainsplus_right);

        gainsleft = _mm256_or_pd(gainsmul_left, gainsplus_left);
        gainsright = _mm256_or_pd(gainsmul_right, gainsplus_right);

        __m256d kmask_left, kmask_right;
        __m256d kmin = {kMinGain, kMinGain, kMinGain, kMinGain};
        kmask_left = _mm256_cmp_pd(gainsleft, kmin, _CMP_LT_OQ);
        kmask_right = _mm256_cmp_pd(gainsright, kmin, _CMP_LT_OQ);
        gainsleft = _mm256_blendv_pd(gainsleft, kmin, kmask_left);
        gainsright = _mm256_blendv_pd(gainsright, kmin, kmask_right);

        //unsort again
        gains0 = _mm256_permute4x64_pd(gainsleft, ymask);
        gains1 = _mm256_permute4x64_pd(gainsright, ymask);

        _mm256_store_pd(gainsdata+twoi, _mm256_unpacklo_pd(gains0, gains1));
        _mm256_store_pd(gainsdata+twoi+4, _mm256_unpackhi_pd(gains0, gains1));

        __m256d momentum_v = {momentum, momentum, momentum, momentum};
        __m256d fourketa = {fourkEta, fourkEta, fourkEta, fourkEta};
        gainsleft = _mm256_mul_pd(fourketa, gainsleft);
        gainsright = _mm256_mul_pd(fourketa, gainsright);
        gainsleft = _mm256_mul_pd(gainsleft, values_left);
        gainsright = _mm256_mul_pd(gainsright, values_right);
        ydeltaleft = _mm256_fmsub_pd(momentum_v, ydeltaleft, gainsleft);
        ydeltaright = _mm256_fmsub_pd(momentum_v, ydeltaright, gainsright);

        ydelta1 = _mm256_permute4x64_pd(ydeltaleft, ymask);
        ydelta2 = _mm256_permute4x64_pd(ydeltaright, ymask);

        _mm256_store_pd(ydeltadata+twoi, _mm256_unpacklo_pd(ydelta1, ydelta2));
        _mm256_store_pd(ydeltadata+twoi+4, _mm256_unpackhi_pd(ydelta1, ydelta2));

        twoi += 8;
    }

    __m256d mean = {0,0,0,0};
    __m256d n_vec = {n,n,n,n};
    __m256d y, ydelta, mean_shuffled, mean_shuffled2;
    int mean_mask = 0b10001101; //setup such that we have [mean1 mean3 mean0 mean2]
    int mean_mask2 = 0b11011000; //setup such that we have [mean0 mean2 mean1 mean3]
    int mean_last_mask = 0b0110;
    int twon = 2*n;
    for(int i=0; i<twon; i+=4){
        //load y, ydelta
        y = _mm256_load_pd(ydata+i);
        ydelta = _mm256_load_pd(ydeltadata+i);
        
        //update step
        y = _mm256_add_pd(y, ydelta);
        mean = _mm256_add_pd(mean, y);

        _mm256_store_pd(ydata+i, y);
    }
    mean_shuffled = _mm256_permute4x64_pd(mean, mean_mask);
    mean_shuffled2 = _mm256_permute4x64_pd(mean, mean_mask2);
    mean = _mm256_hadd_pd(mean_shuffled2, mean_shuffled); //mean now holds [mean0 mean1 mean1 mean0]
    mean = _mm256_permute_pd(mean, mean_last_mask);
    // take mean
    mean = _mm256_div_pd(mean, n_vec);
    // center
    for (int i = 0; i < twon; i+=4){
        y = _mm256_load_pd(ydata+i);
        y = _mm256_sub_pd(y, mean);
        _mm256_store_pd(ydata+i, y);
    }
}