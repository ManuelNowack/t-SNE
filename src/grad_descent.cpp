#include <tsne/grad_descent.h>
#include <tsne/baseline.h>

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
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
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
    calc_affinities(Y, &var->Q, &var->Q_numerators, &var->D);
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

    double mean0 = 0, mean1 = 0;
    for(int i=0; i<(2*n); i+=2){
        // calculate gains, according to adaptive heuristic of Python implementation
        double ydeltadatai = ydeltadata[i];
        double ydeltadataip1 = ydeltadata[i+1];
        double gradydatai = gradydata[i];
        double gradydataip1 = gradydata[i+1];

        bool positive_grad0 = (gradydatai > 0);
        bool positive_delta0 = (ydeltadatai > 0);
        bool positive_grad1 = (gradydataip1 > 0);
        bool positive_delta1 = (ydeltadataip1 > 0);
        double val0 = gainsdata[i];
        double val1 = gainsdata[i+1];

        val0 = (positive_grad0 == positive_delta0) ? val0 * 0.8 : val0 + 0.2;
        val1 = (positive_grad1 == positive_delta1) ? val1 * 0.8 : val1 + 0.2;
        if (val0 < kMinGain){
            val0 = kMinGain;
        }
        if (val1 < kMinGain){
            val1 = kMinGain;
        }

        gainsdata[i] = val0;
        gainsdata[i+1] = val1;

        //update step
        double v0 = momentum * ydeltadatai - kEta * val0 * gradydatai;
        double v1 = momentum * ydeltadataip1 - kEta * val1 * gradydataip1;
        ydeltadata[i] = v0;
        ydeltadata[i + 1] = v1;
        ydata[i] += v0;
        ydata[i+1] += v1;
        mean0 += ydata[i];
        mean1 += ydata[i+1];
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