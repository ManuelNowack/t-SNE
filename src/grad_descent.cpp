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
