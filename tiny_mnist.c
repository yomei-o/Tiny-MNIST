#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>

#ifdef _MSC_VER
#pragma warning( disable : 4996 )
#pragma warning( disable : 4819 )
#endif

#if defined(_WIN32) && !defined(__GNUC__)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#endif

#define TRAIN_IMAGE "train-images-idx3-ubyte"
#define TRAIN_LABEL "train-labels-idx1-ubyte"
#define TEST_IMAGE "t10k-images-idx3-ubyte"
#define TEST_LABEL "t10k-labels-idx1-ubyte"

#define image_pos(a,b) (a->data+b*784)
#define label_pos(a,b) (a->data+b*10)
#define at(o,a,b) o->data[a*o->cols+b] 

#define max_(a,b) (((a) > (b)) ? (a) : (b))
#define min_(a,b) (((a) < (b)) ? (a) : (b))


// struct 
struct tensor{
	float* data;
	int cols;
	int rows;
};

static struct tensor*  create_tensor(int rows, int cols) {
	struct tensor* ret;
	ret = malloc(sizeof(struct tensor));
	ret->data = malloc(sizeof(float)*rows*cols);
	memset(ret->data, 0, sizeof(float)*rows*cols);
	ret->cols = cols; ret->rows = rows;
	return ret;
}
static void free_tensor(struct tensor* t)
{
	if (t && t->data)free(t->data);
	if (t)free(t);
}
static struct tensor*  copy_tensor(struct tensor*a) {
	struct tensor* ret;
	ret = malloc(sizeof(struct tensor));
	ret->data = malloc(sizeof(float)*a->rows*a->cols);
	memcpy(ret->data,a->data,sizeof(float)*a->rows*a->cols);
	ret->cols = a->cols; ret->rows = a->rows;
	return ret;
}
static struct tensor*  create_rand_tensor(int rows, int cols) {
	struct tensor* ret;
	int i, j;
	ret = create_tensor(rows,cols);
	for (i = 0; i<rows; i++) {
		for (j = 0; j < cols; j++) {
			at(ret, i, j) = (float)(((double)rand()/ RAND_MAX-0.5)/2);
		}
	}
	return ret;
}


// loading
static int buf2int(char* buf_){
	int ret;
	unsigned char* buf = (unsigned char*)buf_;

	ret = buf[0]; ret <<= 8; ret |= buf[1]; ret <<= 8;
	ret |= buf[2]; ret <<= 8; ret |= buf[3];
	return ret;
}

static struct tensor* load_image_file(const char* fn)
{
	struct tensor* ret = NULL;
	FILE* fp;
	int sz, t, w, h, n, i, j;
	char buf[4];

	fp = fopen(fn, "rb");
	if (fp == NULL)goto end;
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	fread(buf, 1, 4, fp);
	t = buf2int(buf);
	if (t != 0x803)goto end;

	fread(buf, 1, 4, fp);
	n = buf2int(buf);
	fread(buf, 1, 4, fp);
	w = buf2int(buf);
	fread(buf, 1, 4, fp);
	h = buf2int(buf);
	if (h*w != 784)goto end;

	ret=create_tensor(n, 784);
	for (i = 0; i < n; i++) {
		for (j = 0; j < 784; j++) {
			fread(buf, 1, 1, fp);
			ret->data[i * 784 + j] = (float)(buf[0] & 255) / 255;
		}
	}
end:
	if (fp)fclose(fp);
	return ret;
}

static struct tensor* load_label_file(const char* fn)
{
	struct tensor* ret = NULL;
	FILE* fp;
	int sz, t, n, i, j;
	char buf[4];

	fp = fopen(fn, "rb");
	if (fp == NULL)goto end;
	fseek(fp, 0, SEEK_END);
	sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	fread(buf, 1, 4, fp);
	t = buf2int(buf);
	if (t != 0x801)goto end;

	fread(buf, 1, 4, fp);
	n = buf2int(buf);
	ret=create_tensor(n, 10);
	for (i = 0; i < n; i++) {
		fread(buf, 1, 1, fp);
		for (j = 0; j < 10; j++) {
			ret->data[i * 10 + j] = (float)((j == buf[0]) ? 1 : 0);
		}
	}
end:
	if (fp)fclose(fp);
	return ret;
}

static void create_batch(int bsz, struct tensor*i_img, struct tensor* i_lbl, struct tensor** o_img, struct tensor** o_lbl)
{
	int asz = i_img->cols*i_img->rows / 784;
	int pos = rand() % asz;
	//int pos = asz - 2;
	int i, off;
	*o_img = create_tensor(bsz, 784);
	*o_lbl = create_tensor(bsz, 10);
	for (i = 0; i < bsz; i++) {
		off = (pos + i) % asz;
		off *= 784;
		memcpy((*o_img)->data + i * 784, i_img->data + off, 784 * sizeof(float));
		off = (pos + i) % asz;
		off *= 10;
		memcpy((*o_lbl)->data + i * 10, i_lbl->data + off, 10 * sizeof(float));
	}
}

// print
static void print_image(float* a)
{
	int w = 28, h = 28, i, j;
	for (j = 0; j < h; j++) {
		for (i = 0; i < w; i++)
		{
			printf("%s", *a == 0 ? "--" : "11");
			a++;
		}
		printf("\n");
	}
	printf("\n");
}

static void print_label(float* a)
{
	int i;
	for (i = 0; i < 10; i++) {
		printf("%d: %f\n", i, a[i]);
	}
	printf("\n");
}

// opration

static struct tensor* transpose(struct tensor* a)
{
	struct tensor* out=NULL;
	out=create_tensor(a->cols, a->rows);
	int r, c;
	for (r = 0; r < a->rows; r++) {
		for (c = 0; c < a->cols; c++) {
			at(out,c, r) = at(a,r, c);
		}
	}
	return out;
}

static struct tensor* dot(struct tensor* a, struct tensor* b)
{
	struct tensor* out = NULL;
	int n = a->cols;
	if (n == b->rows) {
		int nrow = a->rows;
		int ncol = b->cols;
		int col, row, i;
		out=create_tensor(nrow, ncol);
		for (col = 0; col < ncol; col++) {
			for (row = 0; row < nrow; row++) {
				for (i = 0; i < n; i++) {
					at(out,row, col) += at(a,row, i) * at(b,i, col);
				}
			}
		}
	}
	return out;
}

static struct tensor* add(struct tensor* a, struct tensor* other)
{
	struct tensor* out = NULL;
	out = copy_tensor(a);

	int i,n = min_(a->cols*a->rows, other->cols*other->rows);
	for (i = 0; i < n; i++) {
		out->data[i] += other->data[i];
	}
	return out;
}

static struct tensor* sub(struct tensor* a, struct tensor* other)
{
	struct tensor* out = NULL;
	out = copy_tensor(a);

	int i,n = min_(a->cols*a->rows, other->cols*other->rows);
	for (i = 0; i < n; i++) {
		out->data[i] -= other->data[i];
	}
	return out;
}
static struct tensor* mul(struct tensor* a, struct tensor* other)
{
	struct tensor* out = NULL;
	out = copy_tensor(a);

	int i,n = min_(a->cols*a->rows, other->cols*other->rows);
	for (i = 0; i < n; i++) {
		out->data[i] *= other->data[i];
	}
	return out;
}

static struct tensor* mul_real(struct tensor* a,float  t)
{
	struct tensor* out = NULL;
	out = copy_tensor(a);

	int  i,n = a->rows*a->cols;
	for (i = 0; i < n; i++) {
		out->data[i] *= t;
	}
	return out;
}

static struct tensor* div_real(struct tensor* a, float  t)
{
	struct tensor* out = NULL;
	out = copy_tensor(a);

	int  i, n = a->rows*a->cols;
	for (i = 0; i < n; i++) {
		out->data[i] /= t;
	}
	return out;
}

static struct tensor* sum(struct tensor* a)
{
	struct tensor*out=NULL;
	int r,c;
	out=create_tensor(1, a->cols);
	for (r = 0; r < a->rows; r++) {
		for (c = 0; c < a->cols; c++) {
			at(out,0, c) += at(a,r, c);
		}
	}
	return out;
}

static float sigmoid_real(float v)
{
	return (float)(1 / (1 + exp(-v)));
}

static struct tensor* sigmoid(struct tensor* a)
{
	struct tensor*out = NULL;
	out=create_tensor(a->rows, a->cols);
	int i, n = a->rows*a->cols;
	for (i = 0; i < n; i++) {
		out->data[i] = sigmoid_real(a->data[i]);
	}
	return out;
}

static struct tensor* sigmoid_grad(struct tensor* a)
{
	struct tensor* out=NULL;
	out=create_tensor(a->rows, a->cols);
	int i, n = a->rows*a->cols;
	for (i = 0; i < n; i++) {
		float v = sigmoid_real(a->data[i]);
		out->data[i] = (1 - v) * v;
	}
	return out;
}

static struct tensor* softmax(struct tensor* a)
{
	struct tensor* out=NULL;
	out=create_tensor(a->rows, a->cols);
	int r,i;
	for (r = 0; r < a->rows; r++) {
		float c = 0;
		for (i = 0; i < a->cols; i++) {
			c = max_(c, at(a,r, i));
		}
		float* exp_a=malloc(sizeof(float)*a->cols);
		float sum_exp_a = 0;
		for (i = 0; i < a->cols; i++) {
			float v = (float)exp(at(a,r, i) - c);
			exp_a[i] = v;
			sum_exp_a += v;
		}
		for (i = 0; i < a->cols; i++) {
			at(out,r, i) = exp_a[i] / sum_exp_a;
		}
		free(exp_a);
	}
	return out;
}

float argmax(struct tensor*a, int row) {
	int i = 0,j;
	for (j = 1; j < a->cols; j++) {
		if (at(a,row, j) > at(a,row, i)) {
			i = j;
		}
	}
	return (float)i;
};

// nn

struct tensor* predict(struct tensor*x, struct tensor* w1, struct tensor* b1, struct tensor*w2, struct tensor*b2)
{
	struct tensor* t1 = dot(x, w1);
	struct tensor* t2 = add(t1, b1);
	struct tensor* t3 = sigmoid(t2);
	struct tensor* t4 = dot(t3,w2);
	struct tensor* t5 = add(t4,b2);
	struct tensor* t6 = softmax(t5);
	free_tensor(t1);
	free_tensor(t2);
	free_tensor(t3);
	free_tensor(t4);
	free_tensor(t5);

	return t6;
}

float accuracy(struct tensor*x,struct tensor*t, struct tensor*w1, struct tensor*b1, struct tensor*w2, struct tensor*b2)
{
	int rows = min_(x->rows, t->rows);
	struct tensor* y = predict(x,w1,b1,w2,b2);
	int acc = 0,row;
	for (row = 0; row < rows; row++) {
		float a = argmax(y, row);
		float b = argmax(t, row);
		if (a == b) {
			acc++;
		}
	}
	free_tensor(y);
	return (float)acc / rows;
}

void gradient(struct tensor*x,struct tensor* t, 
	struct tensor*w1, struct tensor*b1, struct tensor*w2, struct tensor*b2,
	struct tensor**w1_, struct tensor**b1_, struct tensor**w2_, struct tensor**b2_)
{
	int batch_num = x->rows;

	struct tensor* t1 = dot(x, w1);
	struct tensor* a1 =add(t1,b1);

	struct tensor* z1 = sigmoid(a1);

	struct tensor* t2 = dot(z1, w2);
	struct tensor* a2 = add(t2,b2);

	struct tensor* y = softmax(a2);

	struct tensor* t3 = sub(y, t);
	struct tensor* dy = div_real(t3,(float)batch_num);

	struct tensor* t4 = transpose(z1);
	*w2_ = dot(t4,dy);
	*b2_=sum(dy);

	struct tensor* w2t = transpose(w2);
	struct tensor* dz1 = dot(dy,w2t);
	struct tensor* t5 = sigmoid_grad(a1);
	struct tensor* da1 = mul(t5,dz1);

	struct tensor* t6 = transpose(x);
	*w1_= dot(t6,da1);
	*b1_= sum(da1);

	free_tensor(t1);
	free_tensor(t2);
	free_tensor(t3);
	free_tensor(t4);
	free_tensor(t5);
	free_tensor(t6);
	free_tensor(a1);
	free_tensor(z1);
	free_tensor(a2);
	free_tensor(y);
	free_tensor(dy);
	free_tensor(w2t);
	free_tensor(dz1);
	free_tensor(da1);
}

// main
int main()
{

#if defined(_WIN32) && !defined(__GNUC__)
	//	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_WNDW);
	//	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_WNDW);
	//	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_WNDW);
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	int input = 28 * 28;
	int hidden = 50;
	int output = 10;
	//int iteration = 10000;
	int iteration = 2000;
	int batch_size = 100;
	float learning_rate = 0.1f;
	int i;

	struct tensor *a,*b, *c, *d, *e,*f,*y;
	struct tensor *w1, *w2, *b1, *b2;
	struct tensor *w1_, *w2_, *b1_, *b2_;
	struct tensor *w1__, *w2__, *b1__, *b2__;
	struct tensor *w1___, *w2___, *b1___, *b2___;
	struct tensor *tw1, *tw2, *tb1, *tb2;

	//iteration = 100;

	a = load_image_file(TRAIN_IMAGE);
	b = load_label_file(TRAIN_LABEL);
	c = load_image_file(TEST_IMAGE);
	d=load_label_file(TEST_LABEL);

	//train
	w1 = create_rand_tensor(input, hidden);
	b1 = create_rand_tensor(1, hidden);
	w2 = create_rand_tensor(hidden,10);
	b2 = create_rand_tensor(1, 10);
	for (i = 0; i < iteration; i++) {
		create_batch(batch_size,a, b, &e, &f);
		gradient(e, f, w1, b1, w2, b2, &w1_,&b1_, &w2_, &b2_);
		if ((i + 1) % 100 == 0) {
			float t = accuracy(e,f,w1,b1,w2,b2);
			printf("[train %d] %f\n", i + 1, t);
		}


		tw1 = w1;tb1 = b1;
		tw2 = w2; tb2 = b2;

		w1__ = mul_real(w1_,learning_rate);
		b1__ = mul_real(b1_,learning_rate);
		w2__ = mul_real(w2_,learning_rate);
		b2__ = mul_real(b2_,learning_rate);
		w1___= sub(w1,w1__);
		b1___ = sub(b1,b1__);
		w2___ = sub(w2,w2__);
		b2___ = sub(b2,b2__);

		w1 = w1___; b1 = b1___;
		w2 = w2___; b2 = b2___;

		free_tensor(tw1);
		free_tensor(w1_);
		free_tensor(w1__);
		free_tensor(tw2);
		free_tensor(w2_);
		free_tensor(w2__);
		free_tensor(tb1);
		free_tensor(b1_);
		free_tensor(b1__);
		free_tensor(tb2);
		free_tensor(b2_);
		free_tensor(b2__);

		free_tensor(e);
		free_tensor(f);
	}

	//predict
	create_batch(1, c, d, &e, &f);
	y=predict(e, w1, b1, w2, b2);
	print_image(e->data);
	for (i = 0; i < 10; i++) {
		printf("%d: %f\n", i, y->data[i]);
	}

	free_tensor(e);
	free_tensor(f);
	free_tensor(y);


	free_tensor(a);
	free_tensor(b);
	free_tensor(c);
	free_tensor(d);

	free_tensor(w1);
	free_tensor(b1);
	free_tensor(w2);
	free_tensor(b2);

	return 0;

}


