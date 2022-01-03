#include <iostream>
#include <algorithm>
using namespace std;
extern "C"
{
   void clean_mesh(float v[], int f[], int v_size, int f_size, int *v_res, int *f_res);
}



int get_fa(int* fa, int x)
{
   return x == fa[x] ? x : (fa[x] = get_fa(fa, fa[x]));
}

int cmp(int x, int y, float *v)
{
   if (v[x*3] < v[y*3]) return true;
   if (v[x*3] > v[y*3]) return false;

   if (v[x*3+1] < v[y*3+1]) return true;
   if (v[x*3+1] > v[y*3+1]) return false;
   
   if (v[x*3+2] > v[y*3+2]) return false;
   return true;
}
void clean_mesh(float v[], int f[], int v_size, int f_size, int *v_res, int *f_res)
{
   int     fa[v_size];
   int    ind[v_size];
//   int    tmp[v_size];
   float   da[v_size];
   float xiao[v_size];
//   for (int i=0;i<v_size;i++)
//   {
//      fa[i]  = i;
//      ind[i] = i;
//   }
//
//   for (int i=1; i<v_size; i<<=1)
//   for (int j=0; j<v_size; j=j+2*i)
//   {
//      if (j+i >= v_size) continue;
//      for (int k=0;k<i;k++) tmp[k] = ind[j+k];
//      int z = 0;   int z_last = i;
//      int x = j;
//      int y = j+i; int y_last = min(j+i+i, v_size);
//      
//      while (z<z_last && y<y_last)
//      if (cmp(tmp[z], ind[y], v)) ind[x++] = tmp[z++]; 
//      else ind[x++] = ind[y++];
//      while (z<z_last) ind[x++] = tmp[z++];
//      while (y<y_last) ind[x++] = ind[y++];
//   }
//
//   /*
//   sort(ind, ind+v_size, [v](int x, int y)->bool{
//         if (v[x*3] < v[y*3]) return true;
//         if (v[x*3] > v[y*3]) return false;
//
//         if (v[x*3+1] < v[y*3+1]) return true;
//         if (v[x*3+1] > v[y*3+1]) return false;
//   
//         if (v[x*3+2] > v[y*3+2]) return false;
//         return true;
//      }
//   );
//   */
//   int pre = 0;
//   for (int i=1; i<v_size; i++)
//   if (v[ind[pre]*3+0]==v[ind[i]*3+0] && v[ind[pre]*3+1]==v[ind[i]*3+1] && v[ind[pre]*3+2]==v[ind[i]*3+2])
//      fa[ind[i]] = ind[pre];
//   else
//      pre = i;
//   pre = 0;
//   for (int i=0;i<v_size;i++)
//   if (fa[i]==i)
//   {
//      v[pre*3+0] = v[i*3+0];
//      v[pre*3+1] = v[i*3+1];
//      v[pre*3+2] = v[i*3+2];
//      ind[i] = pre;
//      pre++;
//   }
//   for (int i=0;i<f_size*3;i++) f[i] = ind[fa[f[i]]];
   // --------------- 合并成功 -------------------------------------------------
   /*
   cout<<pre*1.0/v_size<<endl;
   (*v_res) = pre;
   (*f_res) = f_size;
   */
   int pre = v_size;
   for (int i=0;i<pre;i++)
   {
      fa[i]   = i;
      ind[i]  = i;
      da[i]   = -100;
      xiao[i] =  100;
   }
   for (int i=0;i<f_size;i++)
   {
      fa[get_fa(fa, f[i*3+0])] = get_fa(fa, f[i*3+1]);
      fa[get_fa(fa, f[i*3+1])] = get_fa(fa, f[i*3+2]);
      fa[get_fa(fa, f[i*3+2])] = get_fa(fa, f[i*3+0]);
   }
   for (int i=0;i<pre;i++)
   {
      int tmp = get_fa(fa, i);
      float val = v[i*3+2];
      da[tmp] = max(da[tmp] , val);
      xiao[tmp] = min(xiao[tmp] , val);
   }

   int index = 0;
   float height = 0;
   for (int i=0;i<pre;i++)
   {
      float tmp = da[i]-xiao[i];
      if (tmp > height)
      {
         index = i;
         height= tmp;
      }
   }

   int cnt = 0;
   for (int i=0;i<pre;i++)
   if (fa[i]==index)
   {
      v[cnt*3+0] = v[i*3+0];
      v[cnt*3+1] = v[i*3+1];
      v[cnt*3+2] = v[i*3+2];
      ind[i] = cnt;
      cnt+=1;
   }
   else
   {
      ind[i] = -1;
   }
   int hhh = 0;
   for (int i=0;i<f_size;i++)
   if (ind[f[i*3]]!=-1)
   {
      f[hhh*3+0] = ind[f[i*3+0]];
      f[hhh*3+1] = ind[f[i*3+1]];
      f[hhh*3+2] = ind[f[i*3+2]];
      hhh+=1;
   }
   (*v_res) = cnt;
   (*f_res) = hhh;
}