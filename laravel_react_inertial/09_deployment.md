Here’s the **normal response** for the topic **"How to deploy a Laravel + React + Inertia.js app?"** along with the suggested file name at the top:

---

### **File Name**: `09_deployment.md`

---

### **How to Deploy a Laravel + React + Inertia.js App**

#### **Goal**
Deploy your application to a production server.

---

#### **Step 1: Build React Assets**
1. Build your React assets for production using Vite:
   ```bash
   npm run build
   ```
2. This generates optimized and minified CSS and JavaScript files in the `public/build` directory.

---

#### **Step 2: Configure Laravel for Production**
1. Set the environment to production in your `.env` file:
   ```env
   APP_ENV=production
   APP_DEBUG=false
   ```
2. Generate a secure application key:
   ```bash
   php artisan key:generate
   ```
3. Optimize the application for production:
   ```bash
   php artisan config:cache
   php artisan route:cache
   php artisan view:cache
   ```

---

#### **Step 3: Prepare the Server**
1. **Forge**:
   - Use Laravel Forge to provision and manage your server.
   - Deploy your application using Forge's deployment script:
     ```bash
     cd /home/forge/your-site.com
     git pull origin main
     composer install --no-interaction --prefer-dist --optimize-autoloader
     npm run build
     php artisan migrate --force
     ```

2. **Vercel**:
   - Use Vercel for frontend deployment.
   - Create a `vercel.json` file in your project root:
     ```json
     {
       "version": 2,
       "builds": [
         {
           "src": "public/index.php",
           "use": "@vercel/php"
         }
       ],
       "routes": [
         {
           "src": "/(.*)",
           "dest": "public/index.php"
         }
       ]
     }
     ```
   - Deploy your application using the Vercel CLI:
     ```bash
     vercel
     ```

3. **AWS**:
   - Use AWS Elastic Beanstalk or EC2 for deployment.
   - Set up a production environment and deploy your application using the AWS CLI or console.

---

#### **Step 4: Set Up the Web Server**
1. **Nginx**:
   - Configure your Nginx server block:
     ```nginx
     server {
         listen 80;
         server_name your-site.com;
         root /home/forge/your-site.com/public;

         index index.php;

         location / {
             try_files $uri $uri/ /index.php?$query_string;
         }

         location ~ \.php$ {
             include snippets/fastcgi-php.conf;
             fastcgi_pass unix:/var/run/php/php8.1-fpm.sock;
         }

         location ~ /\.ht {
             deny all;
         }
     }
     ```

2. **Apache**:
   - Configure your Apache virtual host:
     ```apache
     <VirtualHost *:80>
         ServerName your-site.com
         DocumentRoot /var/www/your-site.com/public

         <Directory /var/www/your-site.com/public>
             AllowOverride All
         </Directory>

         ErrorLog ${APACHE_LOG_DIR}/error.log
         CustomLog ${APACHE_LOG_DIR}/access.log combined
     </VirtualHost>
     ```

---

#### **Step 5: Deploy the Application**
1. Push your code to a Git repository (e.g., GitHub, GitLab, Bitbucket).
2. Use a deployment tool (e.g., Forge, Vercel, AWS) to deploy your application.
3. Run any necessary migrations:
   ```bash
   php artisan migrate --force
   ```

---

#### **Step 6: Verify the Deployment**
1. Visit your application in the browser to ensure it is working correctly.
2. Check the server logs for any errors:
   ```bash
   tail -f /var/log/nginx/error.log
   ```

---

#### **Key Takeaways**
- Build React assets using `npm run build`.
- Configure Laravel for production by setting `APP_ENV=production` and optimizing the application.
- Deploy to a server like Forge, Vercel, or AWS.
- Set up the web server (Nginx or Apache) to serve your application.

---

#### **Next Steps**
- Learn how to test a Laravel + React + Inertia.js app (see `10_testing.md`).
- Explore how to implement real-time features with Laravel Echo and WebSockets (see `11_realtime_features.md`).

---

Let me know if you need further assistance or the content for the next topic!