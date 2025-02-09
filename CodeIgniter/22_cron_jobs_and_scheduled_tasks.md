## **How to Set Up Cron Jobs and Automate Tasks in CodeIgniter?**  

Cron jobs in CodeIgniter allow you to automate tasks like **sending emails, clearing logs, generating reports, or updating databases** at scheduled intervals. This guide will show how to **set up cron jobs** and automate tasks in CodeIgniter.  

---

## **1. Create a Controller for Cron Jobs**  
📁 **`app/Controllers/Cron.php`**  

```php
namespace App\Controllers;
use CodeIgniter\Controller;

class Cron extends Controller {

    public function sendEmailReminders() {
        $email = \Config\Services::email();
        
        // Configure email
        $email->setTo('user@example.com');
        $email->setFrom('admin@example.com', 'Admin');
        $email->setSubject('Reminder');
        $email->setMessage('This is your scheduled email reminder.');

        if ($email->send()) {
            echo "Email sent successfully!";
        } else {
            echo "Email sending failed.";
        }
    }

    public function clearLogs() {
        $logPath = WRITEPATH . 'logs/';
        array_map('unlink', glob("$logPath*.log"));
        echo "Logs cleared!";
    }
}
```
✅ **Why?**  
- Defines **cron job tasks** inside a dedicated controller.  
- Includes functions for **sending emails and clearing logs**.  

---

## **2. Set Up a Cron Job in Linux**  
### **Step 1: Find Your PHP Binary Path**  
Run:  
```shell
which php
```
Example output:  
```
/usr/bin/php
```

### **Step 2: Add Cron Job Entry**  
Run:  
```shell
crontab -e
```
Add this line to schedule the job every day at **12 AM**:  
```shell
0 0 * * * /usr/bin/php /var/www/html/codeigniter_project/public/index.php cron sendEmailReminders >> /var/www/html/codeigniter_project/logs/cron.log 2>&1
```
✅ **Why?**  
- Runs `sendEmailReminders` function **daily at midnight**.  
- Logs output to `cron.log`.  

---

## **3. Set Up Cron Job in Windows (Task Scheduler)**  
1️⃣ Open **Task Scheduler** → Click **Create Basic Task**.  
2️⃣ Choose **Daily** → Set time (e.g., 12:00 AM).  
3️⃣ In **Action**, select **Start a Program** → Browse **php.exe**.  
4️⃣ In **Arguments**, add:  
   ```
   C:\xampp\htdocs\codeigniter_project\public\index.php cron sendEmailReminders
   ```

✅ **Why?**  
- Automates tasks on Windows using **Task Scheduler**.  

---

## **4. Test Cron Job Manually**  
Run in the browser:  
```
http://localhost:8080/cron/sendEmailReminders
```
OR run in the terminal:  
```shell
php public/index.php cron sendEmailReminders
```

---

## **Conclusion**  
✔ **Cron jobs** automate tasks like **email notifications, log cleanup, and database updates**.  
✔ Works on **Linux (crontab) and Windows (Task Scheduler)**.  
✔ Easily testable via **browser or CLI**.  

🚀 Now your CodeIgniter app is set up for automated tasks! 🎯